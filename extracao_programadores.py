"""
Módulo 10 - Extração de Manuais de Programadores (CarProg, etc.)
Versão simplificada: força 2 colunas e usa OCR simples com clustering.
"""

import os
import re
import cv2
import numpy as np
import pytesseract
from collections import defaultdict
from sklearn.cluster import KMeans
from logger_erros import logger

DEBUG_M10 = os.environ.get('DEBUG_M10', 'false').lower() == 'true'
DEBUG_DIR = os.path.join(os.getcwd(), 'debug_m10')
if DEBUG_M10 and not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR)

KEYWORDS_PINO = ['pin', 'pino', 'terminal', 'no.', 'nº']
KEYWORDS_FUNCAO = ['função', 'funcao', 'function', 'descrição', 'description', 'signal', 'sinal', 'name']

def log_detalhado(msg, nivel='info'):
    extra = {'modulo': 'M10'}
    if nivel == 'info':
        logger.info(f"[M10] {msg}", extra=extra)
    elif nivel == 'warning':
        logger.warning(f"[M10] {msg}", extra=extra)
    elif nivel == 'debug':
        logger.debug(f"[M10] {msg}", extra=extra)
    else:
        logger.info(f"[M10] {msg}", extra=extra)

def salvar_imagem_debug(nome, imagem):
    if DEBUG_M10:
        caminho = os.path.join(DEBUG_DIR, nome)
        cv2.imwrite(caminho, imagem)
        log_detalhado(f"Imagem de debug salva: {caminho}", 'debug')

def preprocessar_imagem(imagem_gray):
    log_detalhado("Aplicando pré-processamento", 'debug')
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(imagem_gray)
    img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if DEBUG_M10:
        salvar_imagem_debug("01_preprocessada.png", img)
    return img

def extrair_tabela_simples(imagem_gray):
    """
    Extrai tabela forçando 2 colunas: Pino e Função.
    Usa OCR simples com clustering de colunas.
    """
    log_detalhado("Iniciando extração simples (2 colunas)", 'info')
    img = preprocessar_imagem(imagem_gray)
    custom_config = r'--oem 3 --psm 6 -l por+eng'
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=custom_config)

    linhas = defaultdict(list)
    for i in range(len(data['text'])):
        txt = data['text'][i].strip()
        if not txt or int(data['conf'][i]) < 30:
            continue
        x = data['left'][i]
        y = data['top'][i]
        linhas[y].append({'texto': txt, 'x': x})

    if not linhas:
        log_detalhado("Nenhum texto extraído", 'warning')
        return {}

    # Agrupar linhas por Y (tolerância 10px)
    grupos = []
    y_ordenadas = sorted(linhas.keys())
    grupo_atual = []
    y_anterior = None
    for y in y_ordenadas:
        if y_anterior is None or abs(y - y_anterior) < 10:
            grupo_atual.extend(linhas[y])
            y_anterior = y
        else:
            grupo_atual.sort(key=lambda item: item['x'])
            grupos.append(grupo_atual)
            grupo_atual = linhas[y]
            y_anterior = y
    if grupo_atual:
        grupo_atual.sort(key=lambda item: item['x'])
        grupos.append(grupo_atual)

    log_detalhado(f"Agrupados {len(grupos)} grupos de linhas", 'debug')

    # Forçar 2 colunas (pino e função) usando clustering
    todas_palavras = [item for grupo in grupos for item in grupo]
    if len(todas_palavras) < 4:
        log_detalhado("Poucas palavras para clustering", 'warning')
        return {}

    X = np.array([[item['x']] for item in todas_palavras])
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    kmeans.fit(X)
    centroides = sorted(kmeans.cluster_centers_.flatten())
    log_detalhado(f"Centroides das colunas: {centroides}", 'debug')

    # Atribuir colunas
    for grupo in grupos:
        for palavra in grupo:
            distancias = [abs(palavra['x'] - col) for col in centroides]
            palavra['coluna'] = np.argmin(distancias)

    # Identificar cabeçalho (primeira linha)
    if not grupos:
        return {}
    cabecalho = grupos[0]
    idx_pino = 0
    idx_funcao = 1
    for i, item in enumerate(cabecalho):
        t = item['texto'].lower()
        if any(k in t for k in KEYWORDS_PINO):
            idx_pino = i
        if any(k in t for k in KEYWORDS_FUNCAO):
            idx_funcao = i

    # Extrair dados
    pin_func = {}
    for grupo in grupos[1:]:
        celulas_por_coluna = {}
        for item in grupo:
            col = item.get('coluna')
            if col is not None:
                celulas_por_coluna[col] = item['texto']
        pino_texto = celulas_por_coluna.get(idx_pino, '')
        func_texto = celulas_por_coluna.get(idx_funcao, '')
        pino_limpo = re.sub(r'[^A-Z0-9]', '', pino_texto.upper())
        if pino_limpo and func_texto:
            pin_func[pino_limpo] = func_texto

    log_detalhado(f"Extraídos {len(pin_func)} pares", 'info')
    return pin_func

def extrair_pares_por_palavras_chave(imagem_gray):
    """Fallback final: extrai pares baseados em padrões como 'BLUE - Reset'."""
    log_detalhado("Iniciando extração por palavras-chave", 'info')
    img = preprocessar_imagem(imagem_gray)
    custom_config = r'--oem 3 --psm 6 -l por+eng'
    texto = pytesseract.image_to_string(img, config=custom_config)

    pin_func = {}
    padrao = re.compile(r'([A-Za-z0-9_]+)\s*[-:]\s*([^\-:\n]+)')
    matches = padrao.findall(texto)
    for chave, valor in matches:
        chave_limpa = chave.strip().upper()
        valor_limpo = valor.strip()
        if chave_limpa and valor_limpo:
            pin_func[chave_limpa] = valor_limpo

    log_detalhado(f"Extraídos {len(pin_func)} pares via palavras-chave", 'info')
    return pin_func

def extrair_programador_com_tesseract(caminho_pdf, limite_paginas=0):
    log_detalhado(f"Iniciando extração M10 para: {caminho_pdf}", 'info')
    try:
        import fitz
    except ImportError:
        log_detalhado("PyMuPDF não instalado", 'error')
        return {}

    doc = fitz.open(caminho_pdf)
    total = len(doc)
    if limite_paginas > 0:
        total = min(total, limite_paginas)

    resultado = {}
    for i in range(total):
        log_detalhado(f"Processando página {i+1}/{total}", 'info')
        page = doc[i]
        pix = page.get_pixmap(dpi=300)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if DEBUG_M10:
            salvar_imagem_debug(f"pagina_{i+1}_original.png", gray)

        # Tenta extração simples (2 colunas)
        parcial = extrair_tabela_simples(gray)
        if not parcial:
            log_detalhado(f"Página {i+1}: extração simples falhou, tentando palavras-chave", 'warning')
            parcial = extrair_pares_por_palavras_chave(gray)

        if parcial:
            log_detalhado(f"Página {i+1}: extraídos {len(parcial)} pares", 'info')
            resultado.update(parcial)
        else:
            log_detalhado(f"Página {i+1}: nenhum dado extraído", 'warning')

    doc.close()
    log_detalhado(f"Extração M10 concluída: total de {len(resultado)} pares", 'info')
    return resultado

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Uso: python extracao_programadores.py <arquivo.pdf>")
        sys.exit(1)
    resultado = extrair_programador_com_tesseract(sys.argv[1])
    for p, f in list(resultado.items())[:20]:
        print(f"{p} -> {f}")
