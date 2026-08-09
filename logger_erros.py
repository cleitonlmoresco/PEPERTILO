import logging
import traceback
import functools
import time
import os
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)

class FiltroContexto(logging.Filter):
    """Garante que os campos 'modulo' e 'funcao' sempre existam para evitar quebra de formatação."""
    def filter(self, record):
        if not hasattr(record, 'modulo'):
            record.modulo = 'geral'
        if not hasattr(record, 'funcao'):
            record.funcao = record.funcName
        return True

FORMATO_ARQUIVO = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(modulo)s | %(funcao)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

FORMATO_CONSOLE = logging.Formatter(
    '%(levelname)-8s | %(modulo)s | %(message)s'
)

def configurar_logger(nome='pipeline', nivel=logging.DEBUG):
    logger = logging.getLogger(nome)
    logger.setLevel(nivel)
    if logger.handlers:
        return logger

    filtro = FiltroContexto()

    arquivo_log = LOG_DIR / f"{nome}_{datetime.now().strftime('%Y%m%d')}.log"
    fh = logging.FileHandler(arquivo_log, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(FORMATO_ARQUIVO)
    fh.addFilter(filtro)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(FORMATO_CONSOLE)
    ch.addFilter(filtro)
    logger.addHandler(ch)

    return logger

logger = configurar_logger('pipeline')

class Severidade:
    BAIXA = 'baixa'
    MEDIA = 'media'
    ALTA = 'alta'
    CRITICA = 'critica'

class ErroPipeline(Exception):
    def __init__(self, mensagem, modulo='desconhecido', severidade=Severidade.ALTA,
                 dados_extra=None, causa=None):
        super().__init__(mensagem)
        self.mensagem = mensagem
        self.modulo = modulo
        self.severidade = severidade
        self.dados_extra = dados_extra or {}
        self.causa = causa
        self.timestamp = datetime.now()

    def to_dict(self):
        return {
            'tipo': type(self).__name__,
            'mensagem': self.mensagem,
            'modulo': self.modulo,
            'severidade': self.severidade,
            'dados_extra': self.dados_extra,
            'causa': str(self.causa) if self.causa else None,
            'timestamp': self.timestamp.isoformat(),
            'traceback': traceback.format_exc() if self.causa else None
        }

class ErroExtracao(ErroPipeline):
    def __init__(self, mensagem, modulo='extracao', **kwargs):
        super().__init__(mensagem, modulo=modulo, **kwargs)

class ErroGrafo(ErroPipeline):
    def __init__(self, mensagem, modulo='grafo', **kwargs):
        super().__init__(mensagem, modulo=modulo, **kwargs)

class ErroOCR(ErroPipeline):
    def __init__(self, mensagem, modulo='ocr', **kwargs):
        super().__init__(mensagem, modulo=modulo, severidade=Severidade.MEDIA, **kwargs)

class ErroExportacao(ErroPipeline):
    def __init__(self, mensagem, modulo='exportacao', **kwargs):
        super().__init__(mensagem, modulo=modulo, **kwargs)

def monitorar(modulo='desconhecido', nivel_log=logging.DEBUG, capturar=True, timer=True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nome_funcao = func.__name__
            extra = {'modulo': modulo, 'funcao': nome_funcao}
            logger.log(nivel_log, f'Iniciando {nome_funcao}', extra=extra)
            inicio = time.perf_counter() if timer else None
            try:
                resultado = func(*args, **kwargs)
                if timer:
                    duracao = time.perf_counter() - inicio
                    logger.log(nivel_log, f'{nome_funcao} concluida em {duracao:.3f}s', extra=extra)
                else:
                    logger.log(nivel_log, f'{nome_funcao} concluida', extra=extra)
                return resultado
            except ErroPipeline as e:
                logger.error(f'{nome_funcao} falhou: {e.mensagem} [severidade={e.severidade}]', extra=extra)
                if capturar:
                    raise
                return None
            except Exception as e:
                msg = f'Erro inesperado em {nome_funcao}: {str(e)}'
                logger.critical(msg, extra=extra)
                logger.debug(traceback.format_exc(), extra=extra)
                if capturar:
                    raise ErroPipeline(mensagem=msg, modulo=modulo, severidade=Severidade.ALTA, causa=e) from e
                return None
        return wrapper
    return decorator

def tratar_erro_controlado(func, *args, valor_padrao=None, modulo='desconhecido', **kwargs):
    try:
        resultado = func(*args, **kwargs)
        return (resultado, None)
    except ErroPipeline as e:
        logger.warning(f'Erro controlado: {e.mensagem} [severidade={e.severidade}]',
                       extra={'modulo': modulo, 'funcao': func.__name__})
        return (valor_padrao, e.to_dict())
    except Exception as e:
        logger.error(f'Erro inesperado controlado: {str(e)}',
                     extra={'modulo': modulo, 'funcao': func.__name__})
        return (valor_padrao, {
            'tipo': type(e).__name__,
            'mensagem': str(e),
            'modulo': modulo,
            'severidade': Severidade.ALTA,
            'traceback': traceback.format_exc()
        })

class ColetorErros:
    def __init__(self):
        self.erros = []
        self.avisos = []

    def adicionar_erro(self, erro, severidade=Severidade.ALTA):
        entrada = erro.to_dict() if isinstance(erro, ErroPipeline) else {
            'mensagem': str(erro), 'severidade': severidade,
            'timestamp': datetime.now().isoformat()
        }
        self.erros.append(entrada)
        if severidade in (Severidade.ALTA, Severidade.CRITICA):
            logger.error(f'[COLETOR] {entrada["mensagem"]}', extra={'modulo': 'Coletor'})
        else:
            logger.warning(f'[COLETOR] {entrada["mensagem"]}', extra={'modulo': 'Coletor'})

    def adicionar_aviso(self, mensagem, dados=None):
        aviso = {'mensagem': mensagem, 'dados': dados or {},
                 'timestamp': datetime.now().isoformat()}
        self.avisos.append(aviso)
        logger.warning(f'[COLETOR] AVISO: {mensagem}', extra={'modulo': 'Coletor'})

    def tem_erros_criticos(self):
        return any(e.get('severidade') in (Severidade.ALTA, Severidade.CRITICA) for e in self.erros)

    def resumo(self):
        return {
            'total_erros': len(self.erros),
            'total_avisos': len(self.avisos),
            'erros': self.erros,
            'avisos': self.avisos
        }
