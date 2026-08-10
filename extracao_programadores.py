"""
Módulo 10 - Extração de Manuais de Programadores (CarProg, etc.)
Usa OCR, detecção de colunas por clustering e agrupamento de linhas.
"""

import re
import cv2
import numpy as np
import pytesseract
from collections import defaultdict
from sklearn.cluster import KMeans
from logger_erros import logger

# Palavras-chave para identificar cabeçalhos de tabela de pinos
KEYWORDS_PINO = ['pin', 'pino', 'terminal', 'no.', 'nº', 'pino']
KEYWORDS_FUNCAO = ['função', 'funcao', 'function', 'descrição', 'description', 'signal', 'sinal', 'name']

def preprocessar_imagem(imagem_gray):
    """Pré-processamento para melhorar OCR: equalização, contraste, binarização."""
    # Equalização de histograma adaptativa (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(imagem_gray)
    # Aumentar contraste
    img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    # Binarização com Otsu
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img

def detectar_colunas_por_clustering(objetos):
    """
    Usa K-means para agrupar objetos por coordenada X (colunas).
    Retorna os centroides dos clusters.
    """
    if len(objetos) < 2:
        return []
    X = np.array([[obj['x']] for obj in objetos])
    # Estima número de colunas como no mínimo 2 e no máximo 5
    n_colunas = min(5, max(2, len(X) // 3))
    kmeans = KMeans(n_clusters=n_colunas, random_state=0, n_init=10)
    kmeans.fit(X)
    centroides = sorted(kmeans.cluster_centers_.flatten())
    return centroides

def extrair_tabela_com_layout_melhorado(imagem_gray):
    """
    Extrai tabela usando análise de layout: detecta células via contornos,
    agrupa por linhas e colunas usando clustering, extrai texto de cada célula.
    """
    if imagem_gray is None:
        return {}

    # Pré-processamento
    img = preprocessar_imagem(imagem_gray)
    # Inverter para fundo preto, texto branco (melhor para contornos)
    img_inv = cv2.bitwise_not(img)

    # Detectar linhas de grade (horizontal e vertical) usando morfologia
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    linhas_h = cv2.morphologyEx(img_inv, cv2.MORPH_OPEN, kernel_h, iterations=2)
    linhas_v = cv2.morphologyEx(img_inv, cv2.MORPH_OPEN, kernel_v, iterations=2)
    grade = cv2.bitwise_or(linhas_h, linhas_v)

    # Encontrar contornos das células (regiões fechadas)
    contornos, _ = cv2.findContours(grade, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    retangulos = []
    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filtra células muito pequenas ou muito grandes
        if w > 15 and h > 10 and w < 800 and h < 150:
            retangulos.append((x, y, w, h))

    if len(retangulos) < 4:
        # Fallback: agrupar textos por posição (sem grade)
        return extrair_tabela_por_ocr_simples(imagem_gray)

    # Extrair texto de cada célula com Tesseract (OCR localizado)
    celulas_texto = {}
    for (x, y, w, h) in retangulos:
        # Recortar a célula
        celula = img[y:y+h, x:x+w]
        # OCR localizado
        config = r'--oem 3 --psm 8 -l por+eng'
        texto = pytesseract.image_to_string(celula, config=config).strip()
        if texto:
            celulas_texto[(x, y)] = texto

    # Agrupar células por linha (Y) e coluna (X)
    # Detectar linhas (agrupar por Y com tolerância 10px)
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

    # Ordenar cada linha por X
    for y in linhas_celulas:
        linhas_celulas[y].sort(key=lambda item: item['x'])

    # Detectar colunas usando clustering em todas as células de todas as linhas
    todas_celulas = [item for linha in linhas_celulas.values() for item in linha]
    if len(todas_celulas) < 2:
        return {}

    centroides_colunas = detectar_colunas_por_clustering(todas_celulas)
    if len(centroides_colunas) < 2:
        return {}

    # Atribuir cada célula a uma coluna baseada no centroide mais próximo
    for y_linha, celulas in linhas_celulas.items():
        for celula in celulas:
            distancias = [abs(celula['x'] - col) for col in centroides_colunas]
            col_idx = np.argmin(distancias)
            celula['coluna'] = col_idx

    # Identificar cabeçalho: pegar a primeira linha que contenha palavras-chave
    cabecalho_encontrado = False
    idx_pino = None
    idx_funcao = None

    linhas_ordenadas = sorted(linhas_celulas.keys())
    for y_linha in linhas_ordenadas:
        celulas = linhas_celulas[y_linha]
        textos = [c['texto'].lower() for c in celulas]
        # Verificar se tem palavras de pino e função
        tem_pino = any(any(k in t for k in KEYWORDS_PINO) for t in textos)
        tem_funcao = any(any(k in t for k in KEYWORDS_FUNCAO) for t in textos)
        if tem_pino and tem_funcao:
            cabecalho_encontrado = True
            # Mapear colunas
            for i, texto in enumerate(textos):
                if any(k in texto.lower() for k in KEYWORDS_PINO):
                    idx_pino = i
                if any(k in texto.lower() for k in KEYWORDS_FUNCAO):
                    idx_funcao = i
            break

    if not cabecalho_encontrado:
        # Fallback: usar as duas primeiras colunas como pino e função
        idx_pino = 0
        idx_funcao = 1 if len(centroides_colunas) > 1 else 0

    # Extrair dados das linhas seguintes
    pin_func = {}
    for y_linha in linhas_ordenadas[1:]:  # pula cabeçalho
        celulas = linhas_celulas[y_linha]
        # Organizar por coluna
        celulas_por_coluna = {}
        for cel in celulas:
            col = cel.get('coluna')
            if col is not None:
                celulas_por_coluna[col] = cel['texto']
        # Pega os textos das colunas identificadas
        pino_texto = celulas_por_coluna.get(idx_pino, '')
        func_texto = celulas_por_coluna.get(idx_funcao, '')
        pino_limpo = re.sub(r'[^A-Z0-9]', '', pino_texto.upper())
        if pino_limpo and func_texto:
            pin_func[pino_limpo] = func_texto

    return pin_func

def extrair_tabela_por_ocr_simples(imagem_gray):
    """Fallback: agrupa textos por linha e coluna (versão melhorada)"""
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

    # Detectar colunas via clustering em todas as palavras
    todas_palavras = [item for grupo in grupos for item in grupo]
    if len(todas_palavras) < 4:
        return {}

    centroides = detectar_colunas_por_clustering(todas_palavras)
    if len(centroides) < 2:
        return {}

    # Atribuir cada palavra a uma coluna
    for grupo in grupos:
        for palavra in grupo:
            distancias = [abs(palavra['x'] - col) for col in centroides]
            palavra['coluna'] = np.argmin(distancias)

    # Identificar cabeçalho (primeira linha)
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
        # Organizar por coluna
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

    return pin_func

def extrair_pares_por_palavras_chave(imagem_gray):
    """Fallback final: extrai pares baseados em padrões como 'BLUE - Reset'"""
    img = preprocessar_imagem(imagem_gray)
    custom_config = r'--oem 3 --psm 6 -l por+eng'
    texto = pytesseract.image_to_string(img, config=custom_config)

    pin_func = {}
    # Padrões: "BLUE - Reset", "RED - +5V", "BKGD: 3.3V"
    padrao = re.compile(r'([A-Za-z0-9_]+)\s*[-:]\s*([^\-:\n]+)')
    matches = padrao.findall(texto)
    for chave, valor in matches:
        chave_limpa = chave.strip().upper()
        valor_limpo = valor.strip()
        if chave_limpa and valor_limpo:
            pin_func[chave_limpa] = valor_limpo

    return pin_func

def extrair_programador_com_tesseract(caminho_pdf, limite_paginas=0):
    """
    Função principal: extrai tabela de pinagem de manuais de programadores.
    """
    try:
        import fitz
    except ImportError:
        logger.error("PyMuPDF não instalado", extra={'modulo': 'M10'})
        return {}

    doc = fitz.open(caminho_pdf)
    total = len(doc)
    if limite_paginas > 0:
        total = min(total, limite_paginas)

    resultado = {}
    for i in range(total):
        page = doc[i]
        # Usar 300 DPI para melhor OCR
        pix = page.get_pixmap(dpi=300)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Tenta extrair com layout melhorado primeiro
        parcial = extrair_tabela_com_layout_melhorado(gray)
        if not parcial:
            # Fallback para OCR simples
            parcial = extrair_tabela_por_ocr_simples(gray)
        if not parcial:
            # Fallback para palavras-chave
            parcial = extrair_pares_por_palavras_chave(gray)

        resultado.update(parcial)
        logger.debug(f"Página {i+1}: {len(parcial)} funções extraídas (M10)", extra={'modulo': 'M10'})

    doc.close()
    return resultado

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Uso: python extracao_programadores.py <arquivo.pdf>")
        sys.exit(1)
    resultado = extrair_programador_com_tesseract(sys.argv[1])
    for p, f in list(resultado.items())[:20]:
        print(f"{p} -> {f}")
