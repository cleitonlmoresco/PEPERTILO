"""
Módulo 6 - Extração de Datasheet.
Extrai a tabela de descrição de pinos de um datasheet (PDF com texto ou imagem)
e retorna um dicionário {pino: função}.
"""

import re
import pdfplumber
import tempfile
import os
from logger_erros import logger, monitorar, ErroOCR, ErroPipeline, Severidade, tratar_erro_controlado, ColetorErros
from utils import to_native

SINONIMOS_PINO = [
    'pin', 'terminal', 'pino', 'pin no', 'pin number', 'nº', 'no.',
    'terminal no', 'connector pin'
]
SINONIMOS_FUNCAO = [
    'função', 'funcao', 'function', 'descrição', 'descricao',
    'description', 'signal', 'sinal', 'nome', 'name', 'designation'
]

MAPA_SIGLAS = {
    'VDD': 'Alimentação positiva',
    'VCC': 'Alimentação',
    'VSS': 'Terra',
    'GND': 'Terra',
    'CAN_H': 'Comunicação CAN High',
    'CAN_L': 'Comunicação CAN Low',
    'CAN H': 'Comunicação CAN High',
    'CAN L': 'Comunicação CAN Low',
    'K-LINE': 'Diagnóstico K-Line',
    'K LINE': 'Diagnóstico K-Line',
    'IGN': 'Ignição (15)',
    'BAT': 'Bateria (30)',
    'BAT+': 'Bateria (30)',
    'KL15': 'Ignição (15)',
    'KL30': 'Bateria (30)',
    'KL31': 'Terra (31)',
    'MISO': 'Comunicação SPI (MISO)',
    'MOSI': 'Comunicação SPI (MOSI)',
    'SCLK': 'Comunicação SPI (Clock)',
    'SCK': 'Comunicação SPI (Clock)',
    'CS': 'Chip Select (SPI)',
    'SDA': 'Comunicação I2C (Dados)',
    'SCL': 'Comunicação I2C (Clock)',
    'TX': 'Transmissão Serial',
    'RX': 'Recepção Serial',
    'RST': 'Reset',
    'RESET': 'Reset',
    'BOOT': 'Bootloader',
    'VPP': 'Tensão de programação',
    'VREF': 'Tensão de referência',
}

RE_PINO_LIMPO = re.compile(r'[^A-Z0-9]')

def normalizar_cabecalho(cabecalho):
    if not cabecalho:
        return ''
    cabecalho = str(cabecalho).lower().strip()
    for char, repl in [('ç', 'c'), ('ã', 'a'), ('õ', 'o'), ('é', 'e'), ('ê', 'e'),
                       ('í', 'i'), ('ó', 'o'), ('ú', 'u'), ('á', 'a'), ('â', 'a')]:
        cabecalho = cabecalho.replace(char, repl)
    return cabecalho

def encontrar_indice_coluna(cabecalho, sinonimos):
    for i, col in enumerate(cabecalho):
        col_norm = normalizar_cabecalho(col)
        if any(sin in col_norm for sin in sinonimos):
            return i
    return None

def limpar_pino(pino_str):
    return RE_PINO_LIMPO.sub('', str(pino_str).strip().upper())

def expandir_siglas(texto):
    if not texto:
        return texto
    for sigla, traducao in sorted(MAPA_SIGLAS.items(), key=lambda x: len(x[0]), reverse=True):
        texto = re.sub(r'\b' + re.escape(sigla) + r'\b', traducao, texto, flags=re.IGNORECASE)
    return texto

def _extrair_com_pdfplumber(caminho_pdf):
    pin_func = {}
    with pdfplumber.open(caminho_pdf) as pdf:
        for page in pdf.pages:
            tabelas = page.extract_tables()
            for tabela in tabelas:
                if not tabela or len(tabela) < 2:
                    continue
                cabecalho = [str(c) if c else '' for c in tabela[0]]
                idx_pino = encontrar_indice_coluna(cabecalho, SINONIMOS_PINO)
                idx_func = encontrar_indice_coluna(cabecalho, SINONIMOS_FUNCAO)
                if idx_pino is not None and idx_func is not None:
                    for linha in tabela[1:]:
                        if linha and idx_pino < len(linha) and idx_func < len(linha):
                            pino = limpar_pino(linha[idx_pino])
                            func = str(linha[idx_func]).strip() if linha[idx_func] else ''
                            if pino and func:
                                pin_func[pino] = func
    return pin_func

def _extrair_com_ocr(caminho_pdf):
    try:
        import fitz
        from paddleocr import PaddleOCR
        doc = fitz.open(caminho_pdf)
        pin_func = {}
        ocr = PaddleOCR(use_angle_cls=True, lang='en')

        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=300)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                img_path = tmp.name
                pix.save(img_path)

            try:
                resultado = ocr.ocr(img_path, cls=True)
                if resultado and resultado[0]:
                    linhas_texto = []
                    for line in resultado[0]:
                        box = line[0]
                        texto = line[1][0]
                        y_medio = (box[0][1] + box[2][1]) / 2
                        x_medio = (box[0][0] + box[2][0]) / 2
                        linhas_texto.append({'texto': texto, 'x': x_medio, 'y': y_medio})

                    linhas_texto.sort(key=lambda l: l['y'])
                    linhas_agrupadas = []
                    grupo_atual = []
                    y_atual = None
                    for l in linhas_texto:
                        if y_atual is None or abs(l['y'] - y_atual) < 10:
                            grupo_atual.append(l)
                            y_atual = l['y'] if y_atual is None else y_atual
                        else:
                            linhas_agrupadas.append(sorted(grupo_atual, key=lambda x: x['x']))
                            grupo_atual = [l]
                            y_atual = l['y']
                    if grupo_atual:
                        linhas_agrupadas.append(sorted(grupo_atual, key=lambda x: x['x']))

                    cabecalho_encontrado = False
                    idx_pino = None
                    idx_func = None
                    for linha in linhas_agrupadas:
                        textos = [l['texto'].strip() for l in linha]
                        if not cabecalho_encontrado:
                            if any('pin' in t.lower() for t in textos) and \
                               any('func' in t.lower() or 'desc' in t.lower() for t in textos):
                                cabecalho_encontrado = True
                                for i, t in enumerate(textos):
                                    if 'pin' in t.lower():
                                        idx_pino = i
                                    if 'func' in t.lower() or 'desc' in t.lower():
                                        idx_func = i
                            continue
                        if idx_pino is not None and idx_func is not None:
                            if len(textos) > max(idx_pino, idx_func):
                                pino = limpar_pino(textos[idx_pino])
                                func = textos[idx_func]
                                if pino and func and len(pino) <= 5:
                                    pin_func[pino] = func
            except Exception as e:
                logger.warning(f"Erro no OCR da página {i+1}: {e}", extra={'modulo': 'M6'})
            finally:
                if os.path.exists(img_path):
                    os.remove(img_path)
        doc.close()
        return pin_func
    except Exception as e:
        raise ErroOCR(f"Falha no OCR: {str(e)}", causa=e)

@monitorar(modulo='M6')
def extrair_datasheet(caminho_pdf, limite_paginas=0):
    """
    Extrai a tabela de pinos de um datasheet.
    Se falhar, tenta o fallback com M9 (Tesseract puro).
    """
    if not caminho_pdf:
        raise ErroPipeline("Caminho do datasheet não informado", modulo='M6', severidade=Severidade.ALTA)

    pin_func = {}
    coletor = ColetorErros()

    # Tenta pdfplumber
    resultado_plumber, erro_plumber = tratar_erro_controlado(
        _extrair_com_pdfplumber, caminho_pdf, valor_padrao={}, modulo='M6'
    )
    if erro_plumber:
        coletor.adicionar_aviso(f"pdfplumber falhou: {erro_plumber.get('mensagem', '')}")
    else:
        pin_func.update(resultado_plumber)

    # Se não extraiu, tenta OCR (PaddleOCR)
    if not pin_func:
        resultado_ocr, erro_ocr = tratar_erro_controlado(
            _extrair_com_ocr, caminho_pdf, valor_padrao={}, modulo='M6'
        )
        if erro_ocr:
            coletor.adicionar_aviso(f"OCR falhou: {erro_ocr.get('mensagem', '')}")
        else:
            pin_func.update(resultado_ocr)

    # FALLBACK FINAL: M9 com Tesseract puro
    if not pin_func:
        try:
            from extracao_manual import extrair_manual_com_tesseract
            logger.info("M6 falhou. Tentando M9 (Tesseract puro)...", extra={'modulo': 'M6'})
            pin_func = extrair_manual_com_tesseract(caminho_pdf, limite_paginas)
        except Exception as e:
            logger.warning(f"M9 também falhou: {e}", extra={'modulo': 'M6'})

    # Expande siglas
    for pino in pin_func:
        pin_func[pino] = expandir_siglas(pin_func[pino])

    if not pin_func:
        logger.warning(f"Nenhuma função extraída do datasheet: {caminho_pdf}", extra={'modulo': 'M6'})

    logger.info(f"Datasheet processado: {len(pin_func)} funções extraídas", extra={'modulo': 'M6'})
    return pin_func

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Uso: python extracao_datasheet.py <datasheet.pdf>")
        sys.exit(1)
    funcoes = extrair_datasheet(sys.argv[1])
    for pino, func in list(funcoes.items())[:20]:
        print(f"  {pino}: {func}")
