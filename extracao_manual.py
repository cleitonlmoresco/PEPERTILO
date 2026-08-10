"""
Módulo 9 - Extração de Manuais e Datasheets usando dados do M4 (OCR já feito).
Recebe os dados por página (textos com coordenadas, retângulos) e extrai
a tabela de pinagem.
"""

import re
from collections import defaultdict
from logger_erros import logger, monitorar

# Palavras-chave para identificar cabeçalho da tabela
KEYWORDS_PINO = ['pin', 'pino', 'terminal', 'no.', 'nº']
KEYWORDS_FUNCAO = ['função', 'funcao', 'function', 'descrição', 'description', 'signal', 'sinal', 'name']

def extrair_tabela_dos_dados(dados_pagina):
    """
    Extrai tabela de pinagem a partir dos dados já extraídos (M4).
    dados_pagina: dict com 'textos', 'retangulos', etc.
    Retorna dict {pino: funcao}
    """
    textos = dados_pagina.get('textos', [])
    retangulos = dados_pagina.get('retangulos', [])

    if not textos:
        return {}

    # Filtrar textos que estão dentro de algum retângulo (possível tabela)
    textos_em_retangulos = []
    for t in textos:
        x, y = t['x'], t['y']
        for rect in retangulos:
            if rect['x0'] <= x <= rect['x1'] and rect['y0'] <= y <= rect['y1']:
                textos_em_retangulos.append(t)
                break

    # Se não houver textos em retângulos, usar todos
    if not textos_em_retangulos:
        textos_em_retangulos = textos

    # Agrupa textos por linha (Y) com tolerância
    linhas = defaultdict(list)
    for t in textos_em_retangulos:
        y = t['y']
        linhas[y].append(t)

    # Ordena grupos de linhas por Y
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

    # Procura cabeçalho (linha com pino e função)
    cabecalho_idx = None
    idx_pino = None
    idx_funcao = None
    for i, grupo in enumerate(grupos):
        textos_linha = [item['texto'].lower() for item in grupo]
        tem_pino = any(any(k in t for k in KEYWORDS_PINO) for t in textos_linha)
        tem_funcao = any(any(k in t for k in KEYWORDS_FUNCAO) for t in textos_linha)
        if tem_pino and tem_funcao:
            cabecalho_idx = i
            # Encontra índices das colunas
            for j, item in enumerate(grupo):
                t = item['texto'].lower()
                if any(k in t for k in KEYWORDS_PINO):
                    idx_pino = j
                if any(k in t for k in KEYWORDS_FUNCAO):
                    idx_funcao = j
            break

    if cabecalho_idx is None or idx_pino is None or idx_funcao is None:
        # Fallback: assume que as duas primeiras colunas são pino e função
        # (apenas se houver mais de 2 colunas na primeira linha)
        if grupos and len(grupos[0]) >= 2:
            idx_pino = 0
            idx_funcao = 1
            cabecalho_idx = 0
        else:
            return {}

    # Extrai dados das linhas abaixo do cabeçalho
    pin_func = {}
    for grupo in grupos[cabecalho_idx+1:]:
        if len(grupo) <= max(idx_pino, idx_funcao):
            continue
        pino_texto = grupo[idx_pino]['texto'].strip()
        func_texto = grupo[idx_funcao]['texto'].strip()
        # Limpa pino (remove não alfanumérico)
        pino_limpo = re.sub(r'[^A-Z0-9]', '', pino_texto.upper())
        # Filtra: pino deve ter entre 1 e 4 caracteres e não ser palavra genérica
        if pino_limpo and len(pino_limpo) <= 4 and pino_limpo not in ['A', 'B', 'C', 'D', 'E', 'F']:
            pin_func[pino_limpo] = func_texto

    return pin_func

@monitorar(modulo='M9')
def extrair_manual_dos_dados(dados_por_pagina):
    """
    Extrai tabelas de pinagem a partir dos dados do M4 (todas as páginas).
    """
    pin_func_global = {}
    for num_pag, dados in dados_por_pagina.items():
        pin_func = extrair_tabela_dos_dados(dados)
        if pin_func:
            logger.info(f"Página {num_pag}: {len(pin_func)} funções extraídas", extra={'modulo': 'M9'})
            pin_func_global.update(pin_func)
    return pin_func_global
