"""
Módulo 2 - Extração Vetorial de PDFs de Engenharia.
Extrai linhas (fios), textos, retângulos (componentes) e círculos (emendas)
de um PDF com camada vetorial, aplicando filtros para ignorar ruídos.
"""

import math
import pdfplumber
import fitz
import numpy as np
from logger_erros import (logger, monitorar, ErroExtracao, ErroPipeline,
                          Severidade, tratar_erro_controlado, ColetorErros)
from utils import to_native

CONFIG = {
    'margem_lateral': 0.05,
    'margem_superior': 0.05,
    'margem_inferior': 0.15,
    'espessura_min': 0.5,
    'espessura_max': 2.5,
    'area_min_componente': 500,
    'tam_min_fonte': 3,
    'dist_max_emenda': 3.0,
}

def ponto_dentro_canvas(x, y, canvas):
    return canvas[0] <= x <= canvas[2] and canvas[1] <= y <= canvas[3]

def cor_valida(cor_rgb):
    if cor_rgb is None:
        return True
    r, g, b = cor_rgb[:3]
    soma = r + g + b
    if soma < 300:
        return True
    if r > 150 and g < 100 and b < 100:
        return True
    if r < 100 and g < 100 and b > 150:
        return True
    return False

@monitorar(modulo='M2')
def extrair_primitivas_vetorial(caminho_pdf):
    if not caminho_pdf or not isinstance(caminho_pdf, str):
        raise ErroExtracao("Caminho do PDF inválido ou não informado", severidade=Severidade.CRITICA)

    coletor = ColetorErros()
    dados = {}

    try:
        pdf = pdfplumber.open(caminho_pdf)
    except Exception as e:
        raise ErroExtracao(f"Não foi possível abrir o PDF: {str(e)}", severidade=Severidade.CRITICA, causa=e)

    try:
        for i, page in enumerate(pdf.pages):
            try:
                canvas = definir_canvas(page)
                linhas = extrair_linhas(page, canvas, coletor)
                textos = extrair_textos(page, canvas)
                retangulos = extrair_retangulos(page, canvas, coletor)
                curvas = extrair_curvas_emendas(page, canvas)

                if not curvas:
                    curvas, erro_curvas = tratar_erro_controlado(
                        extrair_emendas_via_fitz, caminho_pdf, i, canvas,
                        valor_padrao=[], modulo='M2'
                    )
                    if erro_curvas:
                        coletor.adicionar_aviso(f"Página {i+1}: fallback de emendas falhou")

                dados[i + 1] = {
                    'linhas': to_native(linhas),
                    'textos': to_native(textos),
                    'retangulos': to_native(retangulos),
                    'curvas': to_native(curvas),
                    'canvas': to_native(canvas),
                    'width': page.width,
                    'height': page.height
                }

                if len(linhas) < 5:
                    coletor.adicionar_aviso(f"Página {i+1}: poucas linhas detectadas ({len(linhas)})")
                if len(textos) < 3:
                    coletor.adicionar_aviso(f"Página {i+1}: poucos textos detectados ({len(textos)})")

            except Exception as e:
                coletor.adicionar_erro(
                    ErroExtracao(f"Erro na página {i+1}: {str(e)}", causa=e),
                    severidade=Severidade.MEDIA
                )

    finally:
        pdf.close()

    if not dados:
        raise ErroExtracao("Nenhuma página foi extraída com sucesso",
                           severidade=Severidade.CRITICA, dados_extra=coletor.resumo())

    if coletor.tem_erros_criticos():
        logger.warning(f"Extração concluída com erros críticos: {coletor.resumo()}", extra={'modulo': 'M2'})

    logger.info(f"Extração concluída: {len(dados)} páginas processadas", extra={'modulo': 'M2'})
    return dados

def definir_canvas(page):
    w, h = page.width, page.height
    return (w * CONFIG['margem_lateral'], h * CONFIG['margem_superior'],
            w * (1 - CONFIG['margem_lateral']), h * (1 - CONFIG['margem_inferior']))

def extrair_linhas(page, canvas, coletor=None):
    linhas_validas = []
    tamanho_canvas = max(canvas[2] - canvas[0], canvas[3] - canvas[1])
    for linha in page.lines:
        try:
            if not ponto_dentro_canvas(linha['x0'], linha['y0'], canvas) and \
               not ponto_dentro_canvas(linha['x1'], linha['y1'], canvas):
                continue
            espessura = linha.get('linewidth', 1)
            if espessura < CONFIG['espessura_min'] or espessura > CONFIG['espessura_max']:
                continue
            if not cor_valida(linha.get('stroking_color')):
                continue
            comp = math.hypot(linha['x1'] - linha['x0'], linha['y1'] - linha['y0'])
            if comp > 0.8 * tamanho_canvas:
                continue
            linhas_validas.append(((linha['x0'], linha['y0']), (linha['x1'], linha['y1'])))
        except Exception as e:
            if coletor:
                coletor.adicionar_aviso(f"Linha ignorada: {str(e)}")
    return linhas_validas

def extrair_textos(page, canvas):
    textos = []
    palavras, erro = tratar_erro_controlado(page.extract_words, keep_blank_chars=False,
                                            valor_padrao=[], modulo='M2')
    for w in palavras:
        xc = (w['x0'] + w['x1']) / 2
        yc = (w['y0'] + w['y1']) / 2
        if not ponto_dentro_canvas(xc, yc, canvas):
            continue
        altura = w.get('height', 0)
        if altura < CONFIG['tam_min_fonte']:
            continue
        texto = w['text'].strip()
        if not texto:
            continue
        textos.append({'x': xc, 'y': yc, 'texto': texto, 'tam': altura})
    return textos

def extrair_retangulos(page, canvas, coletor=None):
    retangulos = []
    for rect in page.rects:
        try:
            cx = (rect['x0'] + rect['x1']) / 2
            cy = (rect['y0'] + rect['y1']) / 2
            if not ponto_dentro_canvas(cx, cy, canvas):
                continue
            area = abs(rect['width'] * rect['height'])
            if area < CONFIG['area_min_componente']:
                continue
            if rect.get('linewidth', 1) == 0 and not rect.get('fill'):
                continue
            retangulos.append({
                'x0': rect['x0'], 'y0': rect['y0'], 'x1': rect['x1'], 'y1': rect['y1'],
                'width': rect['width'], 'height': rect['height'], 'area': area
            })
        except Exception as e:
            if coletor:
                coletor.adicionar_aviso(f"Retângulo ignorado: {str(e)}")
    return retangulos

def extrair_curvas_emendas(page, canvas):
    emendas = []
    if hasattr(page, 'curves') and page.curves:
        for curve in page.curves:
            try:
                if not ponto_dentro_canvas((curve['x0']+curve['x1'])/2, (curve['y0']+curve['y1'])/2, canvas):
                    continue
                w = abs(curve['x1'] - curve['x0'])
                h = abs(curve['y1'] - curve['y0'])
                if 2 < w < 8 and 2 < h < 8 and abs(w - h) < 3:
                    if curve.get('fill'):
                        emendas.append(((curve['x0']+curve['x1'])/2, (curve['y0']+curve['y1'])/2))
            except Exception:
                continue
    return emendas

def extrair_emendas_via_fitz(caminho_pdf, num_pagina, canvas):
    emendas = []
    doc = fitz.open(caminho_pdf)
    try:
        if num_pagina < len(doc):
            pagina = doc[num_pagina]
            paths = pagina.get_drawings()
            for path in paths:
                pontos = []
                for item in path.get('items', []):
                    if item[0] == 'l':
                        pontos.append(item[1])
                        pontos.append(item[2])
                if len(pontos) < 6:
                    continue
                xs = [p.x for p in pontos]
                ys = [p.y for p in pontos]
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)
                w, h = x1 - x0, y1 - y0
                if 2 < w < 8 and 2 < h < 8 and abs(w - h) < 3:
                    if path.get('fill') and path['fill'] != (1.0, 1.0, 1.0):
                        cx, cy = (x0+x1)/2, (y0+y1)/2
                        if ponto_dentro_canvas(cx, cy, canvas):
                            emendas.append((cx, cy))
    except Exception as e:
        logger.warning(f"Fallback fitz falhou: {e}", extra={'modulo': 'M2'})
    finally:
        doc.close()
    return emendas

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Uso: python extracao_vetorial.py <arquivo.pdf>")
        sys.exit(1)
    try:
        resultado = extrair_primitivas_vetorial(sys.argv[1])
        for pag, dados in resultado.items():
            print(f"Página {pag}: {len(dados['linhas'])} linhas, {len(dados['textos'])} textos")
    except ErroPipeline as e:
        logger.error(f"Falha: {e.to_dict()}")
        sys.exit(1)
