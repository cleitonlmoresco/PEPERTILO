"""
Módulo 10 - Extração de Manuais de Programadores (CarProg, etc.)
SEM dependências externas (não usa sklearn).
Usa histograma de posições X para detectar colunas.
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
        log_detalhado(f"Imagem salva: {caminho}", 'debug')

def preprocessar_imagem(imagem_gray):
    log_detalhado("Pré-processamento (CLAHE + contraste + Otsu)", 'debug')
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(imagem_gray)
    img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if DEBUG_M10:
        salvar_imagem_debug("01_preprocessada.png", img)
    return img

def detectar_colunas_por_histograma(objetos, n_colunas=2):
    """
    Detecta colunas usando histograma das posições X.
    Retorna os limites das colunas (centroides).
    """
    if len(objetos) < 2:
        return []
    xs = [obj['x'] for obj in objetos]
    # Usa percentis para dividir em n_colunas grupos
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    if n < 2:
        return []
    # Tenta dividir em 2 colunas por padrão
    if n_colunas == 2:
        mediana = np.percentile(xs_sorted, 50)
        return [np.mean([x for x in xs if x < mediana]), np.mean([x for x in xs if x >= mediana])]
    # Para mais colunas, usa quartis ou percentis
    elif n_colunas == 3:
        p1 = np.percentile(xs_sorted, 33)
        p2 = np.percentile(xs_sorted, 66)
        grupos = [
            [x for x in xs if x < p1],
            [x for x in xs if p1 <= x < p2],
            [x for x in xs if x >= p2]
        ]
        return [np.mean(g) for g in grupos if g]
    else:
        # Fallback para 2 colunas
        mediana = np.percentile(xs_sorted, 50)
        return [np.mean([x for x in xs if x < mediana]), np.mean([x for x in xs if x >= mediana])]

def extrair_tabela_com_layout_melhorado(imagem_gray):
    log_detalhado("Iniciando extração com layout (detecção de grade)", 'info')
    if imagem_gray is None:
        log_detalhado("Imagem vazia", 'warning')
        return {}

    img = preprocessar_imagem(imagem_gray)
    img_inv = cv2.bitwise_not(img)

    # Detectar linhas de grade (morfologia)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    linhas_h = cv2.morphologyEx(img_inv, cv2.MORPH_OPEN, kernel_h, iterations=2)
    linhas_v = cv2.morphologyEx(img_inv, cv2.MORPH_OPEN, kernel_v, iterations=2)
    grade = cv2.bitwise_or(linhas_h, linhas_v)
    if DEBUG_M10:
        salvar_imagem_debug("02_grade.png", grade)

    contornos, _ = cv2.findContours(grade, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    retangulos = []
    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 15 and h > 10 and w < 800 and h < 150:
            retangulos.append((x, y, w, h))
    log_detalhado(f"Detectados {len(retangulos)} retângulos", 'debug')

    if len(retangulos) < 4:
        log_detalhado("Poucos retângulos, fallback para OCR simples", 'warning')
        return extrair_tabela_por_ocr_simples(imagem_gray)

    # Extrair texto de cada célula
    celulas_texto = {}
    for idx, (x, y, w, h) in enumerate(retangulos):
        celula = img[y:y+h, x:x+w]
        if DEBUG_M10:
            salvar_imagem_debug(f"03_celula_{idx}_{x}_{y}.png", celula)
        config = r'--oem 3 --psm 8 -l por+eng'
        texto = pytesseract.image_to_string(celula, config=config).strip()
        if texto:
            celulas_texto[(x, y)] = texto

    log_detalhado(f"Extraídos {len(celulas_texto)} textos de células", 'debug')
    if not celulas_texto:
        return {}

    # Agrupar por linhas (Y)
    linhas_celulas = defaultdict(list)
    for (x, y), texto in celulas_texto.items():
        adicionado = False
        for y_linha in list(linhas_celulas.keys()):
            if abs(y - y_linha) < 10:
                linhas_celulas[y_linha].append({'x': x, 'y': y, 'texto': texto})
                adicionado = True
                break
        if not adicionado:
            linhas_celulas[y].append({'x': x, 'y': y, 'texto': texto})

    for y in linhas_celulas:
        linhas_celulas[y].sort(key=lambda item: item['x'])

    log_detalhado(f"Agrupadas {len(linhas_celulas)} linhas", 'debug')

    # Detectar colunas usando histograma (sem sklearn)
    todas_celulas = [item for linha in linhas_celulas.values() for item in linha]
    if len(todas_celulas) < 2:
        return {}

    centroides = detectar_colunas_por_histograma(todas_celulas, n_colunas=2)
    if len(centroides) < 2:
        log_detalhado("Menos de 2 colunas, fallback", 'warning')
        return extrair_tabela_por_ocr_simples(imagem_gray)

    log_detalhado(f"Centroides das colunas: {centroides}", 'debug')

    # Atribuir cada célula a uma coluna
    for y_linha, celulas in linhas_celulas.items():
        for celula in celulas:
            distancias = [abs(celula['x'] - col) for col in centroides]
            col_idx = np.argmin(distancias)
            celula['coluna'] = col_idx

    # Identificar cabeçalho
    cabecalho_encontrado = False
    idx_pino = None
    idx_funcao = None

    linhas_ordenadas = sorted(linhas_celulas.keys())
    for y_linha in linhas_ordenadas:
        celulas = linhas_celulas[y_linha]
        textos = [c['texto'].lower() for c in celulas]
        tem_pino = any(any(k in t for k in KEYWORDS_PINO) for t in textos)
        tem_funcao = any(any(k in t for k in KEYWORDS_FUNCAO) for t in textos)
        if tem_pino and tem_funcao:
            cabecalho_encontrado = True
            for i, texto in enumerate(textos):
                if any(k in texto.lower() for k in KEYWORDS_PINO):
                    idx_pino = i
                if any(k in texto.lower() for k in KEYWORDS_FUNCAO):
                    idx_funcao = i
            log_detalhado(f"Cabeçalho na linha Y={y_linha}, pino col={idx_pino}, função col={idx_funcao}", 'debug')
            break

    if not cabecalho_encontrado:
        log_detalhado("Cabeçalho não encontrado, usando colunas 0 e 1", 'warning')
        idx_pino = 0
        idx_funcao = 1 if len(centroides) > 1 else 0

    pin_func = {}
    for y_linha in linhas_ordenadas[1:]:
        celulas = linhas_celulas[y_linha]
        celulas_por_coluna = {}
        for cel in celulas:
            col = cel.get('coluna')
            if col is not None:
                celulas_por_coluna[col] = cel['texto']
        pino_texto = celulas_por_coluna.get(idx_pino, '')
        func_texto = celulas_por_coluna.get(idx_funcao, '')
        pino_limpo = re.sub(r'[^A-Z0-9]', '', pino_texto.upper())
        if pino_limpo and func_texto:
            pin_func[pino_limpo] = func_texto

    log_detalhado(f"Extraídos {len(pin_func)} pares via layout", 'info')
    if DEBUG_M10:
        for k, v in list(pin_func.items())[:5]:
            log_detalhado(f"  Exemplo: {k} -> {v}", 'debug')
    return pin_func

def extrair_tabela_por_ocr_simples(imagem_gray):
    log_detalhado("Extraindo por OCR simples (agrupamento coordenadas)", 'info')
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
        return {}

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

    todas_palavras = [item for grupo in grupos for item in grupo]
    if len(todas_palavras) < 4:
        return {}

    centroides = detectar_colunas_por_histograma(todas_palavras, n_colunas=2)
    if len(centroides) < 2:
        return extrair_pares_por_palavras_chave(imagem_gray)

    for grupo in grupos:
        for palavra in grupo:
            distancias = [abs(palavra['x'] - col) for col in centroides]
            palavra['coluna'] = np.argmin(distancias)

    if not grupos:
        return {}
    cabecalho = grupos[0]
    idx_pino = 0
    idx_funcao = 1 if len(centroides) > 1 else 0
    for i, item in enumerate(cabecalho):
        t = item['texto'].lower()
        if any(k in t for k in KEYWORDS_PINO):
            idx_pino = i
        if any(k in t for k in KEYWORDS_FUNCAO):
            idx_funcao = i

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

    log_detalhado(f"Extraídos {len(pin_func)} pares via OCR simples", 'info')
    return pin_func

def extrair_pares_por_palavras_chave(imagem_gray):
    log_detalhado("Extraindo por palavras-chave (padrões)", 'info')
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
    log_detalhado(f"Iniciando M10 para: {caminho_pdf}", 'info')
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
        log_detalhado(f"Página {i+1}/{total}", 'info')
        page = doc[i]
        pix = page.get_pixmap(dpi=300)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if DEBUG_M10:
            salvar_imagem_debug(f"pagina_{i+1}_original.png", gray)

        parcial = extrair_tabela_com_layout_melhorado(gray)
        if not parcial:
            log_detalhado(f"Página {i+1}: layout falhou, tentando OCR simples", 'warning')
            parcial = extrair_tabela_por_ocr_simples(gray)
        if not parcial:
            log_detalhado(f"Página {i+1}: OCR simples falhou, tentando palavras-chave", 'warning')
            parcial = extrair_pares_por_palavras_chave(gray)

        if parcial:
            log_detalhado(f"Página {i+1}: extraídos {len(parcial)} pares", 'info')
            resultado.update(parcial)
        else:
            log_detalhado(f"Página {i+1}: nenhum dado", 'warning')

    doc.close()
    log_detalhado(f"M10 concluído: total {len(resultado)} pares", 'info')
    return resultado

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Uso: python extracao_programadores.py <arquivo.pdf>")
        sys.exit(1)
    resultado = extrair_programador_com_tesseract(sys.argv[1])
    for p, f in list(resultado.items())[:20]:
        print(f"{p} -> {f}")
