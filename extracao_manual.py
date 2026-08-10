"""
Módulo 9 - Extração de Manuais e Datasheets com Tesseract Puro.
Detecta tabelas de pinagem em documentos escaneados.
"""

import re
import cv2
import numpy as np
import pytesseract
from collections import defaultdict
from logger_erros import logger, monitorar, ErroPipeline, Severidade
from utils import to_native

# Palavras-chave para identificar cabeçalhos de tabela de pinos
KEYWORDS_PINO = ['pin', 'pino', 'terminal', 'no.', 'nº']
KEYWORDS_FUNCAO = ['função', 'funcao', 'function', 'descrição', 'description', 'signal', 'sinal', 'name']

def extrair_tabela_por_ocr(imagem_array):
    """
    Extrai tabelas de pinagem de uma imagem usando Tesseract.
    Retorna dict {pino: funcao}
    """
    if imagem_array is None:
        return {}

    # Aplica threshold para melhorar OCR
    _, img = cv2.threshold(imagem_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Configuração do Tesseract: português + inglês
    custom_config = r'--oem 3 --psm 6 -l por+eng'
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=custom_config)

    # Agrupa por linha (Y)
    linhas = defaultdict(list)
    for i in range(len(data['text'])):
        txt = data['text'][i].strip()
        if not txt:
            continue
        conf = int(data['conf'][i])
        if conf < 30:
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
            'h': h,
            'conf': conf
        })

    # Ordena as linhas por Y e agrupa as próximas (tolerância de 10px)
    grupos = []
    y_ordenadas = sorted(linhas.keys())
    grupo_atual = []
    y_anterior = None
    for y in y_ordenadas:
        if y_anterior is None or abs(y - y_anterior) < 10:
            grupo_atual.extend(linhas[y])
            y_anterior = y
        else:
            # Ordena o grupo atual por X
            grupo_atual.sort(key=lambda item: item['x'])
            grupos.append(grupo_atual)
            grupo_atual = linhas[y]
            y_anterior = y
    if grupo_atual:
        grupo_atual.sort(key=lambda item: item['x'])
        grupos.append(grupo_atual)

    # Procura cabeçalho: "Pino" e "Função" na mesma linha ou próximas
    cabecalho_encontrado = False
    idx_pino = None
    idx_funcao = None

    for grupo in grupos:
        textos = [item['texto'].lower() for item in grupo]
        tem_pino = any(any(k in t for k in KEYWORDS_PINO) for t in textos)
        tem_funcao = any(any(k in t for k in KEYWORDS_FUNCAO) for t in textos)
        if tem_pino and tem_funcao:
            cabecalho_encontrado = True
            for i, item in enumerate(grupo):
                t = item['texto'].lower()
                if any(k in t for k in KEYWORDS_PINO):
                    idx_pino = i
                if any(k in t for k in KEYWORDS_FUNCAO):
                    idx_funcao = i
            break

    if not cabecalho_encontrado or idx_pino is None or idx_funcao is None:
        # Fallback: assume que a primeira coluna é pino e a segunda função
        for grupo in grupos:
            if len(grupo) >= 2:
                idx_pino = 0
                idx_funcao = 1
                cabecalho_encontrado = True
                break

    if not cabecalho_encontrado:
        return {}

    # Extrai os dados
    pin_func = {}
    for grupo in grupos:
        if len(grupo) <= max(idx_pino, idx_funcao):
            continue
        pino_texto = grupo[idx_pino]['texto'].strip()
        func_texto = grupo[idx_funcao]['texto'].strip()
        pino_limpo = re.sub(r'[^A-Z0-9]', '', pino_texto.upper())
        if pino_limpo and func_texto:
            pin_func[pino_limpo] = func_texto

    return pin_func

@monitorar(modulo='M9')
def extrair_manual_com_tesseract(caminho_pdf, limite_paginas=0):
    """
    Extrai tabelas de pinagem de um PDF manual usando Tesseract em cada página.
    """
    try:
        import fitz
    except ImportError:
        raise ErroPipeline("PyMuPDF (fitz) não instalado", modulo='M9', severidade=Severidade.CRITICA)

    doc = fitz.open(caminho_pdf)
    total = len(doc)
    if limite_paginas > 0:
        total = min(total, limite_paginas)

    pin_func_global = {}

    for i in range(total):
        page = doc[i]
        pix = page.get_pixmap(dpi=200)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        pin_func = extrair_tabela_por_ocr(gray)
        pin_func_global.update(pin_func)

        logger.debug(f"Página {i+1}: {len(pin_func)} funções extraídas", extra={'modulo': 'M9'})

    doc.close()
    return pin_func_global

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Uso: python extracao_manual.py <arquivo.pdf>")
        sys.exit(1)
    resultado = extrair_manual_com_tesseract(sys.argv[1])
    for p, f in list(resultado.items())[:10]:
        print(f"{p} -> {f}")
