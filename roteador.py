"""
Roteador - Módulo Central de Encaminhamento.
Classifica o arquivo e encaminha para os módulos de processamento adequados.
"""

import os
import sys
import traceback
import cv2
import numpy as np
from classificador import classificar_arquivo, TIPOS_VALIDOS
from logger_erros import logger, monitorar, ErroPipeline, Severidade

# Módulos do pipeline
from extracao_vetorial import extrair_primitivas_vetorial
from grafo_rastreador import processar_diagrama_multipagina, processar_diagrama
from restauracao_img import restaurar_imagem
from deteccao_simbolos import processar_imagem_restaurada
from modo_mpu import processar_modo_mpu


# ============================================================
# SANITIZAÇÃO DEFINITIVA (elimina numpy types)
# ============================================================
def to_native(obj, depth=0):
    """Converte QUALQUER objeto numpy para tipo Python nativo."""
    if depth > 100:
        return str(obj)

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.str_):
        return str(obj)
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return to_native(obj.item(), depth + 1)
        return [to_native(x, depth + 1) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {to_native(k, depth + 1): to_native(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(x, depth + 1) for x in obj]
    if isinstance(obj, tuple):
        return tuple(to_native(x, depth + 1) for x in obj)
    if hasattr(obj, 'item') and callable(obj.item):
        try:
            return to_native(obj.item(), depth + 1)
        except:
            pass
    return obj


# ============================================================
# PROCESSAR PDF VETORIAL
# ============================================================
def processar_pdf_vetorial(caminho):
    """Processa PDF vetorial usando M2 + M5."""
    logger.info(f"Processando PDF vetorial: {caminho}", extra={'modulo': 'Roteador'})
    try:
        dados = extrair_primitivas_vetorial(caminho)
        if not dados:
            return {'status': 'erro', 'mensagem': 'Nenhum dado extraído do PDF.'}

        dados = to_native(dados)

        if len(dados) > 1:
            conexoes, G, pinos, perifs = processar_diagrama_multipagina(dados)
        else:
            conexoes, G, pinos, perifs = processar_diagrama(dados[1])

        return {
            'status': 'ok',
            'modulo': 'M5',
            'arquivo': str(caminho),
            'conexoes': to_native(conexoes),
            'num_conexoes': int(len(conexoes)),
            'num_pinos_ecu': int(len(pinos)),
            'num_paginas': int(len(dados))
        }
    except Exception as e:
        logger.error(f"Erro no processamento vetorial: {e}\n{traceback.format_exc()}", extra={'modulo': 'Roteador'})
        return {'status': 'erro', 'mensagem': str(e)}


# ============================================================
# PROCESSAR IMAGEM LIMPA
# ============================================================
def processar_imagem_limpa(caminho_imagem):
    """Processa imagem limpa: M3 + M4 + M5."""
    logger.info(f"Processando imagem limpa: {caminho_imagem}", extra={'modulo': 'Roteador'})
    try:
        img = cv2.imread(caminho_imagem)
        if img is None:
            return {'status': 'erro', 'mensagem': 'Não foi possível abrir a imagem.'}

        esqueleto, binaria = restaurar_imagem(img)
        dados_pagina = processar_imagem_restaurada(esqueleto, binaria, original=img)
        dados_pagina = to_native(dados_pagina)
        conexoes, G, pinos, perifs = processar_diagrama(dados_pagina)

        return {
            'status': 'ok',
            'modulo': 'M5',
            'conexoes': to_native(conexoes),
            'num_conexoes': int(len(conexoes)),
            'num_pinos': int(len(pinos))
        }
    except Exception as e:
        logger.error(f"Erro no processamento de imagem: {e}\n{traceback.format_exc()}", extra={'modulo': 'Roteador'})
        return {'status': 'erro', 'mensagem': str(e)}


# ============================================================
# PROCESSAR PDF RASTERIZADO (com detecção automática de MPU)
# ============================================================
def processar_pdf_rasterizado(caminho_pdf):
    """Processa PDF rasterizado com detecção automática de Modo MPU."""
    logger.info(">>> INICIANDO PROCESSAMENTO DO PDF...", extra={'modulo': 'Roteador'})
    try:
        import fitz
        doc = fitz.open(caminho_pdf)
        dados_paginas = {}
        resultados_mpu = []

        total_paginas = len(doc)
        if total_paginas == 0:
            doc.close()
            return {'status': 'erro', 'mensagem': 'PDF não contém páginas.'}

        for i, page in enumerate(doc):
            try:
                # Converter página para imagem
                pix = page.get_pixmap(dpi=300)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # Restauração e detecção de símbolos
                esqueleto, binaria = restaurar_imagem(img)
                dados_pagina = processar_imagem_restaurada(esqueleto, binaria, original=img)

                # Sanitização total
                dados_pagina = to_native(dados_pagina)
                dados_paginas[i + 1] = dados_pagina

                logger.info(
                    f">>> PÁGINA {i+1}: {len(dados_pagina.get('textos',[]))} textos, "
                    f"{len(dados_pagina.get('retangulos',[]))} retângulos, "
                    f"{len(dados_pagina.get('linhas',[]))} linhas",
                    extra={'modulo': 'Roteador'}
                )

                # Verificar Modo MPU
                resultado_mpu = processar_modo_mpu(dados_pagina, ferramenta='CarProg_A10')
                resultado_mpu = to_native(resultado_mpu)

                if resultado_mpu.get('modo') == 'mpu':
                    logger.info(">>> MODO MPU ATIVADO – LEITURA DE MICROCONTROLADOR <<<", extra={'modulo': 'MPU'})
                    logger.info(f">>> {resultado_mpu.get('num_pinos', 0)} pinos funcionais encontrados", extra={'modulo': 'MPU'})
                    resultados_mpu.append({
                        'pagina': int(i + 1),
                        'resultado': resultado_mpu
                    })

            except Exception as e:
                logger.error(f"Erro na página {i+1}: {e}\n{traceback.format_exc()}", extra={'modulo': 'Roteador'})
                continue

        doc.close()
        logger.info(f">>> PDF PROCESSADO: {len(dados_paginas)} páginas", extra={'modulo': 'Roteador'})

        # Se alguma página ativou o modo MPU, retornar resultado especial
        if resultados_mpu:
            todos_pinos = []
            for r in resultados_mpu:
                for pino in r['resultado'].get('pinos', []):
                    todos_pinos.append({
                        'pino': int(pino.get('pino', 0)),
                        'lado': str(pino.get('lado', '')),
                        'funcao': str(pino.get('funcao', '')),
                        'texto_original': str(pino.get('texto_original', '')),
                        'cor_fio': str(pino.get('cor_fio', 'N/C')),
                        'ferramenta': str(pino.get('ferramenta', 'CarProg_A10')),
                        'confianca': int(pino.get('confianca', 0))
                    })

            return {
                'status': 'ok',
                'modulo': 'MPU',
                'modo': 'mpu',
                'pinos': todos_pinos,
                'num_pinos': int(len(todos_pinos)),
                'num_paginas': int(len(resultados_mpu)),
                'ferramenta': 'CarProg_A10'
            }

        # Processamento normal (grafo)
        if not dados_paginas:
            return {'status': 'erro', 'mensagem': 'Nenhuma página processada'}

        if len(dados_paginas) > 1:
            conexoes, G, pinos, perifs = processar_diagrama_multipagina(dados_paginas)
        else:
            primeira_pagina = next(iter(dados_paginas.values()))
            conexoes, G, pinos, perifs = processar_diagrama(primeira_pagina)

        return {
            'status': 'ok',
            'modulo': 'M5',
            'conexoes': to_native(conexoes),
            'num_conexoes': int(len(conexoes)),
            'num_paginas': int(len(dados_paginas))
        }

    except Exception as e:
        logger.error(f"Erro no processamento rasterizado: {e}\n{traceback.format_exc()}", extra={'modulo': 'Roteador'})
        return {'status': 'erro', 'mensagem': str(e)}


# ============================================================
# DEMAIS ROTAS
# ============================================================
def processar_foto_celular(caminho_imagem):
    """Processa foto de celular: fallback para processamento de imagem."""
    logger.info(f"Processando foto de celular: {caminho_imagem}", extra={'modulo': 'Roteador'})
    return processar_imagem_limpa(caminho_imagem)


def processar_desconhecido(caminho):
    """Fallback para tipos não reconhecidos."""
    logger.warning(f"Tipo desconhecido: {caminho}", extra={'modulo': 'Roteador'})
    return {'status': 'erro', 'mensagem': 'Tipo de arquivo não suportado.'}


ROTAS = {
    'pdf_vetorial': processar_pdf_vetorial,
    'pdf_rasterizado': processar_pdf_rasterizado,
    'imagem_limpa': processar_imagem_limpa,
    'foto_celular': processar_foto_celular,
    'desconhecido': processar_desconhecido,
}


# ============================================================
# FUNÇÃO PRINCIPAL DO ROTEADOR
# ============================================================
@monitorar(modulo='Roteador')
def rotear_arquivo(caminho_arquivo):
    """
    Classifica o arquivo e o encaminha para o módulo apropriado.
    """
    if not os.path.exists(caminho_arquivo):
        raise ErroPipeline(
            f"Arquivo não encontrado: {caminho_arquivo}",
            modulo='Roteador',
            severidade=Severidade.CRITICA
        )

    tipo = classificar_arquivo(caminho_arquivo)
    descricao = TIPOS_VALIDOS.get(tipo, 'Desconhecido')

    logger.info(f"Arquivo classificado como: {tipo} ({descricao})", extra={'modulo': 'Roteador'})

    funcao = ROTAS.get(tipo, processar_desconhecido)
    resultado = funcao(caminho_arquivo)

    return {
        'arquivo': str(caminho_arquivo),
        'tipo': str(tipo),
        'descricao': str(descricao),
        'resultado': to_native(resultado)
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Uso: python roteador.py <arquivo>")
        sys.exit(1)

    try:
        resultado = rotear_arquivo(sys.argv[1])
        print(f"\nTipo: {resultado['tipo']} - {resultado['descricao']}")
        if resultado['resultado']['status'] == 'ok':
            if resultado['resultado'].get('modo') == 'mpu':
                print(f"Pinos MPU encontrados: {resultado['resultado']['num_pinos']}")
            else:
                print(f"Conexões encontradas: {resultado['resultado'].get('num_conexoes', 0)}")
        else:
            print(f"Erro: {resultado['resultado'].get('mensagem', 'Desconhecido')}")
    except ErroPipeline as e:
        logger.critical(f"Falha crítica: {e.to_dict()}")
        sys.exit(1)