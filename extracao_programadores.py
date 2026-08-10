
"""
Módulo 10 - Extração de Manuais de Programadores (CarProg, etc.)
Usa OCR e análise de layout para extrair tabelas de pinagem.
"""

import re
import cv2
import numpy as np
import pytesseract
from collections import defaultdict
from logger_erros import logger

# Palavras-chave para identificar cabeçalhos de tabela de pinos
KEYWORDS_PINO = ['pin', 'pino', 'terminal', 'no.', 'nº']
KEYWORDS_FUNCAO = ['função', 'funcao', 'function', 'descrição', 'description', 'signal', 'sinal', 'name', 'função']
KEYWORDS_COR = ['color', 'cor', 'wire', 'fio']

def extrair_tabela_com_layout(imagem_gray):
    """
    Extrai tabela usando análise de layout: detecta linhas horizontais/verticais,
    agrupa células e extrai texto de cada célula com Tesseract.
    """
    if imagem_gray is None:
        return {}

    # 1. Binarização
    _, bin = cv2.threshold(imagem_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Inverter para ter fundo preto e texto branco (melhor para contornos)
    bin_inv = cv2.bitwise_not(bin)

    # 2. Detecção de linhas de grade (horizontal e vertical)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    linhas_h = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, kernel_h, iterations=2)
    linhas_v = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, kernel_v, iterations=2)
    grade = cv2.bitwise_or(linhas_h, linhas_v)

    # 3. Encontrar contornos das células (regiões fechadas)
    contornos, _ = cv2.findContours(grade, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    retangulos = []
    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filtra células muito pequenas ou muito grandes
        if w > 20 and h > 15 and w < 1000 and h < 200:
            retangulos.append((x, y, w, h))

    if not retangulos:
        # Fallback: agrupar textos por posição (como no M9)
        return extrair_tabela_por_ocr_simples(imagem_gray)

    # 4. Ordenar retângulos por linha (Y) e coluna (X)
    # Agrupar por Y (tolerância de 10px)
    linhas_celulas = defaultdict(list)
    for x, y, w, h in retangulos:
        # Encontrar linha (agrupar por Y)
        adicionado = False
        for y_linha in list(linhas_celulas.keys()):
            if abs(y - y_linha) < 10:
                linhas_celulas[y_linha].append((x, y, w, h))
                adicionado = True
                break
        if not adicionado:
            linhas_celulas[y].append((x, y, w, h))

    # Ordenar cada linha por X
    for y in linhas_celulas:
        linhas_celulas[y].sort(key=lambda item: item[0])

    # 5. Para cada célula, extrair texto com Tesseract (OCR localizado)
    celulas_texto = {}
    for y_linha, celulas in linhas_celulas.items():
        for x, y, w, h in celulas:
            # Recortar a célula
            celula = bin[y:y+h, x:x+w]
            # Aplicar OCR na célula
            config = r'--oem 3 --psm 8 -l por+eng'
            texto = pytesseract.image_to_string(celula, config=config).strip()
            if texto:
                celulas_texto[(x, y)] = texto

    # 6. Identificar cabeçalho e colunas
    # Pegar a primeira linha com células (que deve ser o cabeçalho)
    y_linhas_ordenadas = sorted(linhas_celulas.keys())
    if not y_linhas_ordenadas:
        return {}

    cabecalho = linhas_celulas[y_linhas_ordenadas[0]]
    cabecalho_textos = []
    for x, y, w, h in cabecalho:
        texto = celulas_texto.get((x, y), '')
        cabecalho_textos.append(texto.lower())

    # Descobrir índices das colunas de pino e função
    idx_pino = None
    idx_funcao = None
    for i, txt in enumerate(cabecalho_textos):
        if any(k in txt for k in KEYWORDS_PINO):
            idx_pino = i
        if any(k in txt for k in KEYWORDS_FUNCAO):
            idx_funcao = i

    # Se não encontrou, assume que a primeira coluna é pino e a segunda função
    if idx_pino is None and len(cabecalho) >= 2:
        idx_pino = 0
        idx_funcao = 1

    if idx_pino is None or idx_funcao is None:
        # Fallback: sem cabeçalho, tenta extrair pares por palavras-chave
        return extrair_pares_por_palavras_chave(imagem_gray)

    # 7. Extrair dados das linhas seguintes
    pin_func = {}
    for y_linha in y_linhas_ordenadas[1:]:  # pula cabeçalho
        celulas = linhas_celulas[y_linha]
        if len(celulas) <= max(idx_pino, idx_funcao):
            continue
        # Pega os textos das células correspondentes
        pino_celula = celulas[idx_pino]
        func_celula = celulas[idx_funcao]
        pino_texto = celulas_texto.get((pino_celula[0], pino_celula[1]), '')
        func_texto = celulas_texto.get((func_celula[0], func_celula[1]), '')
        pino_limpo = re.sub(r'[^A-Z0-9]', '', pino_texto.upper())
        if pino_limpo and func_texto:
            pin_func[pino_limpo] = func_texto

    return pin_func

def extrair_tabela_por_ocr_simples(imagem_gray):
    """Fallback: agrupa textos por linha e coluna (versão melhorada do M9)"""
    _, img = cv2.threshold(imagem_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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

    # Identificar cabeçalho e colunas
    cabecalho = None
    for grupo in grupos:
        textos = [item['texto'].lower() for item in grupo]
        if any(any(k in t for k in KEYWORDS_PINO) for t in textos) and \
           any(any(k in t for k in KEYWORDS_FUNCAO) for t in textos):
            cabecalho = grupo
            break
    if not cabecalho and len(grupos) > 0:
        cabecalho = grupos[0]  # assume primeira linha

    if not cabecalho or len(cabecalho) < 2:
        # Fallback final: extrair pares por palavras-chave
        return extrair_pares_por_palavras_chave(imagem_gray)

    idx_pino = 0
    idx_funcao = 1
    for i, item in enumerate(cabecalho):
        t = item['texto'].lower()
        if any(k in t for k in KEYWORDS_PINO):
            idx_pino = i
        if any(k in t for k in KEYWORDS_FUNCAO):
            idx_funcao = i

    pin_func = {}
    for grupo in grupos[1:]:
        if len(grupo) <= max(idx_pino, idx_funcao):
            continue
        pino = re.sub(r'[^A-Z0-9]', '', grupo[idx_pino]['texto'].upper())
        func = grupo[idx_funcao]['texto'].strip()
        if pino and func:
            pin_func[pino] = func

    return pin_func

def extrair_pares_por_palavras_chave(imagem_gray):
    """Extrai pares chave-valor baseados em padrões como 'BLUE - Reset' ou 'BKGD: 3.3V'"""
    _, img = cv2.threshold(imagem_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    custom_config = r'--oem 3 --psm 6 -l por+eng'
    texto = pytesseract.image_to_string(img, config=custom_config)

    pin_func = {}
    # Padrões: "BLUE - Reset", "RED - +5V", "BKGD: 3.3V", "GND - Terra", etc.
    padrao = re.compile(r'([A-Za-z0-9_]+)\s*[-:]\s*([^\-:\n]+)')
    matches = padrao.findall(texto)
    for chave, valor in matches:
        chave_limpa = chave.strip().upper()
        valor_limpo = valor.strip()
        if chave_limpa and valor_limpo:
            pin_func[chave_limpa] = valor_limpo

    # Se não encontrou, tenta capturar linhas que contenham palavras-chave
    if not pin_func:
        linhas = texto.split('\n')
        for linha in linhas:
            linha = linha.strip()
            if not linha:
                continue
            # Procura por palavras como VDD, GND, RESET, etc.
            for sigla in ['VDD', 'GND', 'RESET', 'BKGD', 'VPP', 'VCC', 'CAN', 'LIN', 'K-LINE']:
                if sigla in linha:
                    partes = linha.split()
                    for i, part in enumerate(partes):
                        if sigla in part:
                            pino = part
                            func = ' '.join(partes[i+1:]) if i+1 < len(partes) else ''
                            if pino and func:
                                pin_func[pino] = func
                            break

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

        # Tenta extrair com layout primeiro
        parcial = extrair_tabela_com_layout(gray)
        if not parcial:
            # Fallback para OCR simples
            parcial = extrair_tabela_por_ocr_simples(gray)
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
