"""
Módulo 10 - Extração de Manuais de Programadores (CarProg, etc.)
Estratégia: detecção de cabeçalho inteligente e alinhamento de colunas por centroides.
"""

import os
import re
import cv2
import numpy as np
import pytesseract
from collections import defaultdict
from logger_erros import logger

# ===== CONFIGURAÇÕES =====
DEBUG_M10 = os.environ.get('DEBUG_M10', 'false').lower() == 'true'
DEBUG_DIR = os.path.join(os.getcwd(), 'debug_m10')
if DEBUG_M10 and not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR)

KEYWORDS_PINO = ['pin', 'pino', 'terminal', 'no.', 'nº']
KEYWORDS_FUNCAO = ['função', 'funcao', 'function', 'descrição', 'description', 'signal', 'sinal', 'name', 'função']

def log(msg, nivel='info'):
    extra = {'modulo': 'M10'}
    getattr(logger, nivel)(f"[M10] {msg}", extra=extra)

def salvar_debug(nome, img):
    if DEBUG_M10:
        caminho = os.path.join(DEBUG_DIR, nome)
        cv2.imwrite(caminho, img)

def preprocessar(imagem_gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(imagem_gray)
    img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if DEBUG_M10:
        salvar_debug("01_preprocessada.png", img)
    return img

def extrair_palavras_por_linha(imagem_gray):
    """Extrai palavras com Tesseract e agrupa por linha (Y)."""
    img = preprocessar(imagem_gray)
    config = r'--oem 3 --psm 6 -l por+eng'
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=config)

    linhas = defaultdict(list)
    for i in range(len(data['text'])):
        txt = data['text'][i].strip()
        if not txt or int(data['conf'][i]) < 30:
            continue
        x = data['left'][i]
        y = data['top'][i]
        w = data['width'][i]
        h = data['height'][i]
        linhas[y].append({
            'texto': txt,
            'x': x,
            'y': y,
            'w': w,
            'h': h
        })

    # Agrupar linhas próximas (tolerância 10px)
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

    return grupos

def identificar_cabecalho(grupos):
    """Identifica a linha de cabeçalho: procura palavras-chave ou a linha com mais palavras."""
    if not grupos:
        return None

    # Primeiro: procura linha com palavras-chave
    for grupo in grupos:
        textos = [item['texto'].lower() for item in grupo]
        tem_pino = any(any(k in t for k in KEYWORDS_PINO) for t in textos)
        tem_funcao = any(any(k in t for k in KEYWORDS_FUNCAO) for t in textos)
        if tem_pino and tem_funcao:
            log(f"Cabeçalho identificado por palavras-chave: {[t['texto'] for t in grupo]}", 'debug')
            return grupo

    # Fallback: linha com mais palavras (>= 2)
    max_palavras = max(len(g) for g in grupos)
    for grupo in grupos:
        if len(grupo) == max_palavras and len(grupo) >= 2:
            log(f"Cabeçalho identificado por maior número de palavras: {[t['texto'] for t in grupo]}", 'debug')
            return grupo

    # Último fallback: primeira linha com 2+ palavras
    for grupo in grupos:
        if len(grupo) >= 2:
            log(f"Cabeçalho identificado como primeira linha com 2+ palavras: {[t['texto'] for t in grupo]}", 'debug')
            return grupo

    return None

def extrair_pares_com_cabecalho(grupos, cabecalho):
    """
    Usa o cabeçalho para definir colunas (centroides X) e extrai pares das linhas seguintes.
    """
    if not cabecalho or len(cabecalho) < 2:
        return {}

    # Calcula centroides das colunas a partir do cabeçalho
    centroides = []
    for item in cabecalho:
        centroides.append(item['x'] + item['w']/2)  # centro da palavra

    # Identifica qual coluna é pino e qual é função (pela posição ou por palavras-chave)
    idx_pino = 0
    idx_funcao = 1
    for i, item in enumerate(cabecalho):
        t = item['texto'].lower()
        if any(k in t for k in KEYWORDS_PINO):
            idx_pino = i
        if any(k in t for k in KEYWORDS_FUNCAO):
            idx_funcao = i

    # Se não encontrou, assume que a primeira coluna é pino e a segunda função
    if idx_pino == idx_funcao:
        idx_pino = 0
        idx_funcao = 1 if len(cabecalho) > 1 else 0

    log(f"Colunas: pino={idx_pino}, função={idx_funcao}, centroides={centroides}", 'debug')

    pin_func = {}
    # Percorre as linhas (exceto a primeira, que é o cabeçalho)
    cabecalho_y = cabecalho[0]['y'] if cabecalho else 0
    for grupo in grupos:
        if not grupo:
            continue
        # Pula linhas que estão muito próximas do cabeçalho (mesma linha)
        if abs(grupo[0]['y'] - cabecalho_y) < 10:
            continue
        # Atribui cada palavra à coluna mais próxima
        colunas = {}
        for item in grupo:
            cx = item['x'] + item['w']/2
            distancias = [abs(cx - c) for c in centroides]
            col_idx = np.argmin(distancias)
            colunas[col_idx] = item['texto']

        pino_texto = colunas.get(idx_pino, '')
        func_texto = colunas.get(idx_funcao, '')
        pino_limpo = re.sub(r'[^A-Z0-9]', '', pino_texto.upper())
        if pino_limpo and func_texto:
            pin_func[pino_limpo] = func_texto

    return pin_func

def extrair_por_palavras_chave(texto_bruto):
    """Fallback final: extrai pares usando padrões como 'BLUE - Reset', 'BROWN - GND'."""
    pin_func = {}
    padrao = re.compile(r'([A-Za-z0-9_]+)\s*[-:]\s*([^\-:\n]+)')
    matches = padrao.findall(texto_bruto)
    for chave, valor in matches:
        chave_limpa = chave.strip().upper()
        valor_limpo = valor.strip()
        if chave_limpa and valor_limpo:
            pin_func[chave_limpa] = valor_limpo

    # Tenta capturar listas de modelos MCU (ex: "MC68HC912B32 9H91F")
    padrao_mcu = re.compile(r'(MC68HC\w+)\s+([A-Z0-9]+)')
    matches_mcu = padrao_mcu.findall(texto_bruto)
    for modelo, mascara in matches_mcu:
        pin_func[modelo] = mascara

    return pin_func

def extrair_programador_com_tesseract(caminho_pdf, limite_paginas=0):
    log(f"Iniciando M10 para: {caminho_pdf}", 'info')
    try:
        import fitz
    except ImportError:
        log("PyMuPDF não instalado", 'error')
        return {}

    doc = fitz.open(caminho_pdf)
    total = len(doc)
    if limite_paginas > 0:
        total = min(total, limite_paginas)

    resultado = {}
    for i in range(total):
        log(f"Página {i+1}/{total}", 'info')
        page = doc[i]
        pix = page.get_pixmap(dpi=300)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if DEBUG_M10:
            salvar_debug(f"pagina_{i+1}_original.png", gray)

        # 1. Extrai palavras agrupadas por linha
        grupos = extrair_palavras_por_linha(gray)
        if not grupos:
            log(f"Página {i+1}: nenhuma palavra extraída", 'warning')
            continue

        # 2. Identifica o cabeçalho
        cabecalho = identificar_cabecalho(grupos)
        if cabecalho:
            log(f"Cabeçalho encontrado: {[t['texto'] for t in cabecalho]}", 'debug')
            pares = extrair_pares_com_cabecalho(grupos, cabecalho)
            if pares:
                resultado.update(pares)
                log(f"Página {i+1}: extraídos {len(pares)} pares", 'info')
                continue

        # 3. Fallback: OCR simples com extração por palavras-chave
        log(f"Página {i+1}: fallback para palavras-chave", 'warning')
        img_proc = preprocessar(gray)
        config = r'--oem 3 --psm 6 -l por+eng'
        texto_bruto = pytesseract.image_to_string(img_proc, config=config)
        pares_fallback = extrair_por_palavras_chave(texto_bruto)
        if pares_fallback:
            resultado.update(pares_fallback)
            log(f"Página {i+1}: extraídos {len(pares_fallback)} pares via fallback", 'info')

    doc.close()
    log(f"M10 concluído: total {len(resultado)} pares", 'info')
    return resultado

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Uso: python extracao_programadores.py <arquivo.pdf>")
        sys.exit(1)
    resultado = extrair_programador_com_tesseract(sys.argv[1])
    for p, f in list(resultado.items())[:20]:
        print(f"{p} -> {f}")
