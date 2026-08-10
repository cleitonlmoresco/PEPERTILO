"""
Módulo 5 - Construção do Grafo Topológico e Rastreamento de Conexões.
Recebe primitivas geométricas e retorna lista de conexões elétricas.
"""

import math
import re
from collections import deque
import networkx as nx
import numpy as np
from logger_erros import logger, monitorar, ErroGrafo, ErroPipeline, Severidade, ColetorErros
from utils import to_native

CONFIG = {
    'dist_max_no': 15.0,
    'dist_texto': 10.0,
    'tol_angulo': 15,
    'dist_emenda': 3.0,
    'altura_fonte_padrao': 8,
}

RE_COR = re.compile(r'^[A-Z]{2}(/[A-Z]{2})?$')
RE_BITOLA = re.compile(r'^\d+\.?\d*\s*mm²$')
RE_PINO = re.compile(r'^[A-Z]\d{1,2}$')
RE_CONTINUACAO = re.compile(r'(?:Pág\.?\s*|Folha\s*|página\s*|continua\s+(?:na|para)?\s*(?:pág\.?|folha)?\s*)(\d+)', re.IGNORECASE)

def ponto_em_retangulo(x, y, rect):
    return rect['x0'] <= x <= rect['x1'] and rect['y0'] <= y <= rect['y1']

def ponto_perto_retangulo(x, y, rect, dist):
    x_min, x_max = rect['x0'] - dist, rect['x1'] + dist
    y_min, y_max = rect['y0'] - dist, rect['y1'] + dist
    return x_min <= x <= x_max and y_min <= y <= y_max

def distancia_ponto_segmento(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    projx = x1 + t * dx
    projy = y1 + t * dy
    return math.hypot(px - projx, py - projy)

def angulo_entre_vetores(v1, v2):
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    m1 = math.hypot(*v1)
    m2 = math.hypot(*v2)
    if m1 == 0 or m2 == 0:
        return 0
    cos = max(-1, min(1, dot / (m1 * m2)))
    return math.degrees(math.acos(cos))

def encontrar_no_mais_proximo(G, pos, max_dist):
    melhor = None
    dist_min = float('inf')
    for no, data in G.nodes(data=True):
        if 'pos' in data:
            d = math.hypot(data['pos'][0] - pos[0], data['pos'][1] - pos[1])
            if d < dist_min and d <= max_dist:
                dist_min = d
                melhor = no
    return melhor

def identificar_ecu(retangulos, canvas):
    if not retangulos:
        return None, []
    cx_canvas = (canvas[0] + canvas[2]) / 2
    cy_canvas = (canvas[1] + canvas[3]) / 2
    def pontuacao(rect):
        area = rect['area']
        cx_rect = (rect['x0'] + rect['x1']) / 2
        cy_rect = (rect['y0'] + rect['y1']) / 2
        dist_centro = math.hypot(cx_rect - cx_canvas, cy_rect - cy_canvas)
        fator = 1 / (1 + dist_centro / max(canvas[2] - canvas[0], canvas[3] - canvas[1]))
        return area * fator
    ordenados = sorted(retangulos, key=pontuacao, reverse=True)
    ecu = ordenados[0]
    perifericos = [r for r in ordenados[1:] if r['area'] > 300]
    return ecu, perifericos

def extrair_pinos_ecu(textos, ecu, altura_fonte):
    pinos = {}
    dist_limite = altura_fonte * 2.5
    for t in textos:
        if RE_PINO.match(t['texto']) and ponto_perto_retangulo(t['x'], t['y'], ecu, dist_limite):
            pinos[t['texto']] = (t['x'], t['y'])
    return pinos

def extrair_perifericos(textos, retangulos, ecu):
    perifericos = {}
    for rect in retangulos:
        if rect == ecu:
            continue
        cx = (rect['x0'] + rect['x1']) / 2
        cy = (rect['y0'] + rect['y1']) / 2
        nome = f"Comp_{cx:.0f}_{cy:.0f}"
        for t in textos:
            if ponto_em_retangulo(t['x'], t['y'], rect):
                nome = t['texto']
                break
        perifericos[nome] = (cx, cy)
    for t in textos:
        if RE_PINO.match(t['texto']):
            continue
        dentro = any(ponto_em_retangulo(t['x'], t['y'], rect) for rect in retangulos)
        if not dentro and len(t['texto']) > 2:
            perifericos[t['texto']] = (t['x'], t['y'])
    return perifericos

def construir_grafo_pagina(linhas, pinos_ecu, perifericos, emendas, config):
    G = nx.Graph()
    for (x1, y1), (x2, y2) in linhas:
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        G.add_node((x1, y1), pos=(x1, y1), tipo='desconhecido')
        G.add_node((x2, y2), pos=(x2, y2), tipo='desconhecido')
        G.add_edge((x1, y1), (x2, y2))
    for pino, pos in pinos_ecu.items():
        no = encontrar_no_mais_proximo(G, pos, config['dist_max_no'])
        if no:
            G.add_node(pino, pos=pos, tipo='pino_ecu', rotulo=pino)
            G.add_edge(pino, no)
        else:
            logger.warning(f"Pino {pino} não conectado ao grafo", extra={'modulo': 'M5'})
    for nome, pos in perifericos.items():
        no = encontrar_no_mais_proximo(G, pos, config['dist_max_no'])
        if no:
            G.add_node(nome, pos=pos, tipo='periferico', rotulo=nome)
            G.add_edge(nome, no)
    classificar_nos(G, emendas, config)
    return G

def classificar_nos(G, emendas, config):
    for no in list(G.nodes()):
        if G.nodes[no].get('tipo') in ['pino_ecu', 'periferico']:
            continue
        grau = G.degree(no)
        if grau == 1:
            G.nodes[no]['tipo'] = 'fim'
            continue
        pos_no = G.nodes[no]['pos']
        if any(math.hypot(pos_no[0] - e[0], pos_no[1] - e[1]) < config['dist_emenda'] for e in emendas):
            G.nodes[no]['tipo'] = 'emenda'
            continue
        if grau == 4:
            vizinhos = list(G.neighbors(no))
            vetores = [(G.nodes[v]['pos'][0] - pos_no[0], G.nodes[v]['pos'][1] - pos_no[1]) for v in vizinhos]
            pares = []
            for i in range(4):
                for j in range(i + 1, 4):
                    ang = angulo_entre_vetores(vetores[i], vetores[j])
                    if abs(ang - 180) < config['tol_angulo']:
                        pares.append((vizinhos[i], vizinhos[j]))
            if len(pares) == 2:
                tipos_viz = [G.nodes[v].get('tipo') for v in vizinhos]
                if any(t in ['pino_ecu', 'periferico'] for t in tipos_viz):
                    G.nodes[no]['tipo'] = 'emenda'
                else:
                    G.nodes[no]['tipo'] = 'cruzamento_sem_conexao'
                    G.nodes[no]['pares'] = pares
        else:
            G.nodes[no]['tipo'] = 'intermediario'

def detectar_continuacoes(G, textos, num_pagina, config):
    for t in textos:
        match = RE_CONTINUACAO.search(t['texto'])
        if match:
            pagina_destino = int(match.group(1))
            no_prox = None
            dist_min = float('inf')
            for no, data in G.nodes(data=True):
                if 'pos' not in data:
                    continue
                d = math.hypot(data['pos'][0] - t['x'], data['pos'][1] - t['y'])
                if d < config['dist_texto'] and d < dist_min:
                    if G.degree(no) == 1 or no_prox is None:
                        dist_min = d
                        no_prox = no
            if no_prox is not None:
                sinal = ''
                for outro in textos:
                    if outro == t:
                        continue
                    if math.hypot(outro['x'] - G.nodes[no_prox]['pos'][0],
                                  outro['y'] - G.nodes[no_prox]['pos'][1]) < config['dist_texto']:
                        if not RE_CONTINUACAO.search(outro['texto']) and not RE_PINO.match(outro['texto']):
                            sinal = outro['texto'].strip()
                            break
                G.nodes[no_prox]['tipo'] = 'continuacao'
                G.nodes[no_prox]['pagina_destino'] = pagina_destino
                G.nodes[no_prox]['pagina_origem'] = num_pagina
                G.nodes[no_prox]['sinal'] = sinal

def unificar_grafos(grafos_pagina):
    G_global = nx.Graph()
    node_map = {}
    for pag, G in grafos_pagina.items():
        for no, data in G.nodes(data=True):
            novo = f"p{pag}_{no}"
            G_global.add_node(novo, **data)
            node_map[(pag, no)] = novo
        for u, v in G.edges():
            G_global.add_edge(node_map[(pag, u)], node_map[(pag, v)])
    continuacoes = []
    for pag, G in grafos_pagina.items():
        for no, data in G.nodes(data=True):
            if data.get('tipo') == 'continuacao':
                continuacoes.append((pag, no, data.get('pagina_destino'), data.get('sinal', '')))
    for pag_orig, no_orig, pag_dest, sinal in continuacoes:
        correspondente = None
        for pag2, G2 in grafos_pagina.items():
            if pag2 != pag_dest:
                continue
            for no2, data2 in G2.nodes(data=True):
                if data2.get('tipo') == 'continuacao' and data2.get('pagina_destino') == pag_orig:
                    if sinal and data2.get('sinal') == sinal:
                        correspondente = (pag2, no2)
                        break
                    elif not sinal or not data2.get('sinal'):
                        correspondente = (pag2, no2)
            if correspondente:
                break
        if correspondente:
            u = node_map[(pag_orig, no_orig)]
            v = node_map[(correspondente[0], correspondente[1])]
            if not G_global.has_edge(u, v):
                G_global.add_edge(u, v)
        else:
            logger.warning(f"Sem correspondência: pág {pag_orig} -> {pag_dest} (sinal: {sinal})", extra={'modulo': 'M5'})
    return G_global

def bfs_rastrear_global(G_global, pinos_iniciais, textos_por_pagina, config):
    conexoes = []
    for pino, no_ini in pinos_iniciais.items():
        if no_ini not in G_global:
            continue
        visitados = set()
        fila = deque([(no_ini, [], [])])
        while fila:
            no, caminho_nos, caminho_arestas = fila.popleft()
            if no in visitados:
                continue
            visitados.add(no)
            if G_global.nodes[no].get('tipo') == 'periferico' and no != no_ini:
                destino = G_global.nodes[no]['rotulo']
                cor = ''
                bitola = ''
                for u, v in caminho_arestas:
                    x1, y1 = G_global.nodes[u]['pos']
                    x2, y2 = G_global.nodes[v]['pos']
                    for pag, textos in textos_por_pagina.items():
                        for t in textos:
                            d = distancia_ponto_segmento(t['x'], t['y'], x1, y1, x2, y2)
                            if d < config['dist_texto']:
                                txt = t['texto']
                                if not cor and RE_COR.match(txt):
                                    cor = txt
                                elif not bitola and RE_BITOLA.match(txt):
                                    bitola = txt
                conexoes.append((pino, destino, cor, bitola))
                continue
            for viz in G_global.neighbors(no):
                if viz not in visitados:
                    if G_global.nodes[no].get('tipo') == 'cruzamento_sem_conexao':
                        pares = G_global.nodes[no].get('pares', [])
                        permitidos = set()
                        for a, b in pares:
                            permitidos.add(a)
                            permitidos.add(b)
                        if viz not in permitidos:
                            continue
                    fila.append((viz, caminho_nos + [no], caminho_arestas + [(no, viz)]))
    return conexoes

@monitorar(modulo='M5')
def processar_diagrama_multipagina(dados_paginas):
    if not dados_paginas:
        raise ErroGrafo("Nenhuma página fornecida", severidade=Severidade.CRITICA)

    coletor = ColetorErros()
    grafos_pagina = {}
    pinos_por_pagina = {}
    textos_por_pagina = {}

    for num_pag, dados in dados_paginas.items():
        try:
            linhas = dados.get('linhas', [])
            textos = dados.get('textos', [])
            retangulos = dados.get('retangulos', [])
            emendas = dados.get('curvas', [])
            canvas = dados.get('canvas', (0, 0, 1000, 1000))

            if len(linhas) < 3:
                coletor.adicionar_aviso(f"Página {num_pag}: poucas linhas ({len(linhas)})")

            ecu, outros = identificar_ecu(retangulos, canvas)
            altura_fonte = np.mean([t['tam'] for t in textos if 'tam' in t]) if textos else CONFIG['altura_fonte_padrao']
            pinos = extrair_pinos_ecu(textos, ecu, altura_fonte) if ecu else {}
            perifs = extrair_perifericos(textos, outros, ecu) if ecu else {}

            G = construir_grafo_pagina(linhas, pinos, perifs, emendas, CONFIG)
            detectar_continuacoes(G, textos, num_pag, CONFIG)

            grafos_pagina[num_pag] = G
            pinos_por_pagina[num_pag] = pinos
            textos_por_pagina[num_pag] = textos

        except Exception as e:
            coletor.adicionar_erro(ErroGrafo(f"Página {num_pag}: {str(e)}", causa=e), severidade=Severidade.MEDIA)

    if not grafos_pagina:
        raise ErroGrafo("Nenhuma página pôde ser processada", dados_extra=coletor.resumo())

    G_global = unificar_grafos(grafos_pagina)
    pinos_iniciais = {}
    for pag, pinos in pinos_por_pagina.items():
        for nome_pino, pos in pinos.items():
            node_global = f"p{pag}_{nome_pino}"
            if node_global in G_global:
                pinos_iniciais[nome_pino] = node_global

    if not pinos_iniciais:
        for no, data in G_global.nodes(data=True):
            if data.get('tipo') == 'pino_ecu':
                pinos_iniciais[data.get('rotulo', no)] = no

    conexoes = bfs_rastrear_global(G_global, pinos_iniciais, textos_por_pagina, CONFIG)
    logger.info(f"Processamento concluído: {len(conexoes)} conexões em {len(grafos_pagina)} páginas", extra={'modulo': 'M5'})
    return conexoes, G_global, pinos_iniciais, {}

def processar_diagrama(dados_pagina):
    return processar_diagrama_multipagina({1: dados_pagina})

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from extracao_vetorial import extrair_primitivas_vetorial
    if len(sys.argv) < 2:
        print("Uso: python grafo_rastreador.py <diagrama.pdf>")
        sys.exit(1)
    try:
        dados = extrair_primitivas_vetorial(sys.argv[1])
        conexoes, G, pinos, perifs = processar_diagrama_multipagina(dados)
        print(f"Conexões: {len(conexoes)}")
        for c in conexoes[:10]:
            print(f"  {c[0]} -> {c[1]}  cor={c[2]}  bitola={c[3]}")
    except ErroPipeline as e:
        logger.error(f"Falha: {e.to_dict()}")
        sys.exit(1)
