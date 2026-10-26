"""
Módulo 2 - Extração Vetorial de PDFs de Engenharia.
Extrai linhas (fios), textos, retângulos (componentes) e círculos (emendas)
de um PDF com camada vetorial, aplicando filtros para ignorar ruídos.
"""

import math
import pdfplumber
import pymupdf as fitz
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

# pdfplumber expõe y0/y1 em coordenadas PDF (origem embaixo), mas 'top'/'bottom'
# já vêm de cima para baixo — que é o sistema usado pelo canvas e pelo fitz.
def _y_topo(obj):
    return obj['top'] if 'top' in obj else obj['y0']

def _y_base(obj):
    return obj['bottom'] if 'bottom' in obj else obj['y1']

def cor_valida(cor_rgb):
    """Aceita preto/cinza-escuro e cores fortes; descarta cinza-claro/fundo.

    pdfplumber pode devolver:
      - None
      - escalar (DeviceGray: 0 = preto, 1 = branco)
      - tupla RGB (0..1) ou CMYK (0..1)
    """
    if cor_rgb is None:
        return True

    if isinstance(cor_rgb, (int, float)):
        return cor_rgb < 0.5

    try:
        valores = list(cor_rgb)
    except TypeError:
        return True

    if not valores:
        return True
    if len(valores) == 1:
        return valores[0] < 0.5
    if len(valores) == 3:
        r, g, b = valores
    elif len(valores) >= 4:
        c, m, y, k = valores[:4]
        r, g, b = (1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k)
    else:
        return True

    # Normaliza 0..1 (pdfplumber) para 0..255 antes dos limiares abaixo.
    if max(r, g, b) <= 1.0:
        r, g, b = r * 255, g * 255, b * 255

    soma = r + g + b
    if soma < 300:
        return True
    if r > 150 and g < 100 and b < 100:
        return True
    if r < 100 and g < 100 and b > 150:
        return True
    return False

@monitorar(modulo='M2')
def extrair_primitivas_vetorial(caminho_pdf, limite_paginas=0):
    if not caminho_pdf or not isinstance(caminho_pdf, str):
        raise ErroExtracao("Caminho do PDF inválido ou não informado", severidade=Severidade.CRITICA)

    coletor = ColetorErros()
    dados = {}

    try:
        pdf = pdfplumber.open(caminho_pdf)
    except Exception as e:
        raise ErroExtracao(f"Não foi possível abrir o PDF: {str(e)}", severidade=Severidade.CRITICA, causa=e) from e

    try:
        for i, page in enumerate(pdf.pages):
            if limite_paginas > 0 and i >= limite_paginas:
                logger.info(f"Limite de {limite_paginas} páginas atingido.", extra={'modulo': 'M2'})
                break
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
            y0, y1 = _y_topo(linha), _y_base(linha)
            if not ponto_dentro_canvas(linha['x0'], y0, canvas) and \
               not ponto_dentro_canvas(linha['x1'], y1, canvas):
                continue
            # linewidth 0 = hairline (largura mínima), muito comum em diagramas CAD
            espessura = linha.get('linewidth') or 1
            if espessura < CONFIG['espessura_min'] or espessura > CONFIG['espessura_max']:
                continue
            if not cor_valida(linha.get('stroking_color')):
                continue
            comp = math.hypot(linha['x1'] - linha['x0'], y1 - y0)
            if comp > 0.8 * tamanho_canvas:
                continue
            linhas_validas.append(((linha['x0'], y0), (linha['x1'], y1)))
        except Exception as e:
            if coletor:
                coletor.adicionar_aviso(f"Linha ignorada: {str(e)}")
    return linhas_validas

def _agrupar_palavras(palavras, gap_max=6.0, tol_linha=2.5):
    """Agrupa palavras vizinhas na mesma linha visual (ex.: 'SENSOR 1').

    extract_words separa rótulos com espaço e o pipeline acabava usando só a
    primeira palavra, longe do fim do fio. extract_text_lines, por outro lado,
    junta rótulos distantes que compartilham a mesma altura. Aqui o critério é
    a distância horizontal real entre as palavras.
    """
    grupos = []
    for w in sorted(palavras, key=lambda p: (p['top'], p['x0'])):
        texto = w.get('text', '').strip()
        if not texto:
            continue
        cy = (w['top'] + w['bottom']) / 2
        for g in grupos:
            if abs(g['cy'] - cy) > tol_linha:
                continue
            if w['x0'] - g['x1'] > gap_max or w['x0'] < g['x0']:
                continue
            g['x1'] = max(g['x1'], w['x1'])
            g['y0'] = min(g['y0'], w['top'])
            g['y1'] = max(g['y1'], w['bottom'])
            g['cy'] = (g['y0'] + g['y1']) / 2
            g['texto'] = f"{g['texto']} {texto}"
            g['tam'] = max(g['tam'], w['bottom'] - w['top'])
            break
        else:
            grupos.append({
                'x0': w['x0'], 'x1': w['x1'],
                'y0': w['top'], 'y1': w['bottom'],
                'cy': cy, 'texto': texto,
                'tam': w['bottom'] - w['top'],
            })
    return grupos

def extrair_textos(page, canvas):
    textos = []
    palavras, erro = tratar_erro_controlado(page.extract_words, keep_blank_chars=False,
                                            valor_padrao=[], modulo='M2')
    for g in _agrupar_palavras(palavras):
        xc = (g['x0'] + g['x1']) / 2
        yc = (g['y0'] + g['y1']) / 2
        if not ponto_dentro_canvas(xc, yc, canvas):
            continue
        altura = g['tam']
        if altura < CONFIG['tam_min_fonte']:
            continue
        textos.append({
            'x': xc, 'y': yc, 'texto': g['texto'], 'tam': altura,
            'x0': g['x0'], 'x1': g['x1'], 'y0': g['y0'], 'y1': g['y1'],
        })
    return textos

def _retangulos_de_curvas(page):
    """Recupera retângulos fechados que o pdfplumber classificou como 'curve'.

    PDFs gerados por CAD/PyMuPDF costumam desenhar caixas como um path de 4
    linhas + fechamento ('h'); nesse caso elas não entram em page.rects e a
    ECU nunca era detectada.
    """
    retangulos = []
    for curve in getattr(page, 'curves', []) or []:
        try:
            path = curve.get('path') or []
            if len(path) < 4 or not any(item[0] == 'h' for item in path):
                continue
            if any(item[0] not in ('m', 'l', 'h') for item in path):
                continue
            pontos = {item[1] for item in path if item[0] in ('m', 'l')}
            if len(pontos) != 4:
                continue
            xs = {round(p[0], 1) for p in pontos}
            ys = {round(p[1], 1) for p in pontos}
            if len(xs) != 2 or len(ys) != 2:
                continue
            retangulos.append(curve)
        except Exception:
            continue
    return retangulos

def extrair_retangulos(page, canvas, coletor=None):
    retangulos = []
    candidatos = list(page.rects) + _retangulos_de_curvas(page)
    vistos = set()
    for rect in candidatos:
        try:
            chave = (round(rect['x0'], 1), round(_y_topo(rect), 1),
                     round(rect['x1'], 1), round(_y_base(rect), 1))
            if chave in vistos:
                continue
            vistos.add(chave)
            cx = (rect['x0'] + rect['x1']) / 2
            cy = (_y_topo(rect) + _y_base(rect)) / 2
            if not ponto_dentro_canvas(cx, cy, canvas):
                continue
            area = abs(rect['width'] * rect['height'])
            if area < CONFIG['area_min_componente']:
                continue
            # linewidth 0 = hairline (comum em CAD); só descarta se não há
            # traço nem preenchimento (objeto invisível).
            if rect.get('linewidth', 1) == 0 and not rect.get('fill') and not rect.get('stroke'):
                continue
            retangulos.append({
                'x0': rect['x0'], 'y0': _y_topo(rect), 'x1': rect['x1'], 'y1': _y_base(rect),
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
                y0, y1 = _y_topo(curve), _y_base(curve)
                cy = (y0 + y1) / 2
                cx = (curve['x0'] + curve['x1']) / 2
                if not ponto_dentro_canvas(cx, cy, canvas):
                    continue
                w = abs(curve['x1'] - curve['x0'])
                h = abs(y1 - y0)
                if 2 < w < 8 and 2 < h < 8 and abs(w - h) < 3:
                    if curve.get('fill'):
                        emendas.append((cx, cy))
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
