"""
Módulo MPU - Modo de Leitura de Microcontrolador.
Detecta automaticamente diagramas de chips e aplica estratégia
de leitura perimetral para mapeamento de pinos.
"""

import re
import numpy as np
from logger_erros import (logger, monitorar, ErroPipeline, ErroExtracao,
                          Severidade, ColetorErros, tratar_erro_controlado)

# ------------------------------------------------------------
# Configurações
# ------------------------------------------------------------
CONFIG_MPU = {
    'area_min_chip': 0.40,
    'max_retangulos': 3,
    'margem_borda': 15,
    'dist_texto_borda': 20,
    'confianca_minima': 50,
}

# ------------------------------------------------------------
# Função sanitizadora
# ------------------------------------------------------------
def to_native(obj, depth=0):
    if depth > 100:
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.str_):
        return str(obj)
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return to_native(obj.item(), depth + 1)
        return [to_native(x, depth + 1) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {to_native(k, depth + 1): to_native(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(x, depth + 1) for x in obj]
    if isinstance(obj, tuple):
        return tuple(to_native(x, depth + 1) for x in obj)
    if hasattr(obj, 'item') and callable(obj.item):
        try:
            return to_native(obj.item(), depth + 1)
        except:
            pass
    return obj


REGEX_PINOS_FUNCIONAIS = re.compile(
    r'\b(RESET|BKGD|VDD|VSS|VDDR|VSSR|VDDX|VSSX|VDDPLL|VSSPLL|'
    r'EXTAL|XTAL|TXD|RXD|CAN|SCI|SPI|MISO|MOSI|SCK|SCL|SDA|'
    r'BOOT|RST|VPP|VREF|VCC|GND|K.?LINE|LIN|IRQ|PD4|PD3|'
    r'\+5V|\+12V|VDD1|VDD2|VSS1|VSS2)\b',
    re.IGNORECASE
)

ALIAS_FUNCOES = {
    'VDD': 'VDD', 'VDDX': 'VDD', 'VDDR': 'VDD', 'VDDPLL': 'VDD',
    'VDD1': 'VDD', 'VDD2': 'VDD', 'VCC': 'VDD', '+5V': 'VDD',
    'VSS': 'GND', 'VSSX': 'GND', 'VSSR': 'GND', 'VSSPLL': 'GND',
    'VSS1': 'GND', 'VSS2': 'GND',
    '+12V': 'VPP',
    'RXD': 'RX', 'TXD': 'TX',
    'K_LINE': 'K-LINE', 'K LINE': 'K-LINE',
    'CAN H': 'CAN_H', 'CAN L': 'CAN_L',
    'PD3': 'BKGD', 'BKGD': 'BKGD',
}

FERRAMENTAS = {
    'CarProg_A10': {
        'RESET': 'AZUL', 'BKGD': 'CINZA',
        'VDD': 'VERMELHO', 'VDDX': 'VERMELHO', 'VDDR': 'VERMELHO',
        'VDDPLL': 'VERMELHO', 'VCC': 'VERMELHO', '+5V': 'VERMELHO',
        'VSS': 'PRETO', 'VSSX': 'PRETO', 'VSSR': 'PRETO',
        'VSSPLL': 'PRETO', 'GND': 'PRETO',
        '+12V': 'VIOLETA', 'VPP': 'VIOLETA',
        'IRQ': 'BRANCO', 'EXTAL': 'BRANCO', 'XTAL': 'BRANCO',
        'PD4': 'LARANJA',
        'RX': 'AMARELO', 'RXD': 'AMARELO',
        'TX': 'VERDE', 'TXD': 'VERDE',
    },
    'UPA_USB': {
        'RESET': 'VERMELHO', 'BKGD': 'AZUL',
        'VDD': 'AMARELO', 'VDDX': 'AMARELO', 'VDDR': 'AMARELO',
        'VDDPLL': 'AMARELO', 'VCC': 'AMARELO', '+5V': 'AMARELO',
        'VSS': 'PRETO', 'VSSX': 'PRETO', 'VSSR': 'PRETO',
        'VSSPLL': 'PRETO', 'GND': 'PRETO',
        'CAN_H': 'BRANCO', 'CAN_L': 'AZUL',
        'K-LINE': 'VERDE', 'K_LINE': 'VERDE',
    },
    'ST10_Flasher': {
        'RESET': 'AZUL', 'BKGD': 'CINZA',
        'VDD': 'VERMELHO', 'VCC': 'VERMELHO', '+5V': 'VERMELHO',
        'VSS': 'PRETO', 'GND': 'PRETO',
        'TXD': 'BRANCO', 'RXD': 'VERDE',
        'BOOT': 'AMARELO', 'TEST': 'CINZA',
        'VPP': 'VERMELHO/PRETO',
    },
    'EEPROM_I2C': {
        'VCC': 'VERMELHO', 'GND': 'PRETO',
        'SDA': 'AZUL', 'SCL': 'BRANCO',
        'WP': 'AMARELO', 'A0': 'CINZA', 'A1': 'BRANCO', 'A2': 'PRETO',
    },
    'EEPROM_SPI': {
        'VCC': 'VERMELHO', 'GND': 'PRETO',
        'CS': 'AMARELO', 'MISO': 'VERDE', 'MOSI': 'AZUL',
        'SCK': 'ROXO', 'WP': 'LARANJA', 'HOLD': 'MARROM',
    }
}


def normalizar_funcao(funcao):
    if not funcao:
        return ''
    return ALIAS_FUNCOES.get(funcao.upper(), funcao.upper())


def obter_cor_fio(funcao, ferramenta_cores):
    if not funcao or not ferramenta_cores:
        return 'N/C'
    if funcao in ferramenta_cores:
        return ferramenta_cores[funcao]
    funcao_norm = normalizar_funcao(funcao)
    if funcao_norm in ferramenta_cores:
        return ferramenta_cores[funcao_norm]
    for chave, cor in ferramenta_cores.items():
        if chave and (chave in funcao or funcao in chave):
            return cor
    return 'N/C'


def determinar_lado_chip(x, y, chip_rect, dist_max):
    dist_topo = abs(y - chip_rect['y0'])
    dist_base = abs(y - chip_rect['y1'])
    dist_esq = abs(x - chip_rect['x0'])
    dist_dir = abs(x - chip_rect['x1'])
    menor = min(dist_topo, dist_base, dist_esq, dist_dir)
    if menor > dist_max:
        return None
    if menor == dist_topo:
        return 'TOPO'
    elif menor == dist_base:
        return 'BASE'
    elif menor == dist_esq:
        return 'ESQUERDA'
    else:
        return 'DIREITA'


def estimar_pino(x, y, lado, chip_rect):
    if lado in ('TOPO', 'BASE'):
        posicao_relativa = (x - chip_rect['x0']) / max(chip_rect['x1'] - chip_rect['x0'], 1)
    else:
        posicao_relativa = (y - chip_rect['y0']) / max(chip_rect['y1'] - chip_rect['y0'], 1)
    posicao_relativa = max(0, min(1, posicao_relativa))
    if lado == 'TOPO':
        return int(posicao_relativa * 30) + 1
    elif lado == 'BASE':
        return int((1 - posicao_relativa) * 30) + 31
    elif lado == 'ESQUERDA':
        return int(posicao_relativa * 20) + 61
    else:
        return int((1 - posicao_relativa) * 20) + 81


@monitorar(modulo='MPU')
def detectar_modo_mpu(dados_pagina):
    coletor = ColetorErros()
    if not dados_pagina:
        coletor.adicionar_aviso("dados_pagina é None ou vazio")
        return False, None

    retangulos = dados_pagina.get('retangulos', [])
    textos = dados_pagina.get('textos', [])
    canvas = dados_pagina.get('canvas', (0, 0, 1000, 1000))

    if len(retangulos) == 0:
        return False, None
    if len(retangulos) > CONFIG_MPU['max_retangulos']:
        return False, None

    maior = max(retangulos, key=lambda r: r['area'])
    area_pagina = (canvas[2] - canvas[0]) * (canvas[3] - canvas[1])
    if area_pagina <= 0:
        return False, None

    proporcao = maior['area'] / area_pagina
    if proporcao < CONFIG_MPU['area_min_chip']:
        return False, None

    textos_perimetro = 0
    textos_interior = 0
    margem = CONFIG_MPU['margem_borda']

    for t in textos:
        try:
            x, y = float(t['x']), float(t['y'])
            perto_borda = (
                abs(x - maior['x0']) < margem or abs(x - maior['x1']) < margem or
                abs(y - maior['y0']) < margem or abs(y - maior['y1']) < margem
            )
            dentro = (maior['x0'] < x < maior['x1'] and maior['y0'] < y < maior['y1'])
            if perto_borda:
                textos_perimetro += 1
            elif dentro:
                textos_interior += 1
        except:
            continue

    if textos_perimetro == 0 and textos_interior == 0:
        return False, None
    if textos_perimetro > textos_interior * 2:
        logger.info(f"Modo MPU detectado: {textos_perimetro} textos no perímetro, {textos_interior} no interior")
        return True, maior
    return False, None


@monitorar(modulo='MPU')
def extrair_pinos_mpu(dados_pagina, chip_rect, ferramenta='CarProg_A10'):
    textos = dados_pagina.get('textos', [])
    if not textos:
        return []

    if ferramenta not in FERRAMENTAS:
        ferramenta = 'CarProg_A10'
    ferramenta_cores = FERRAMENTAS[ferramenta]
    pinos_encontrados = []
    dist_max = CONFIG_MPU['dist_texto_borda']

    chip_x0 = float(chip_rect['x0'])
    chip_y0 = float(chip_rect['y0'])
    chip_x1 = float(chip_rect['x1'])
    chip_y1 = float(chip_rect['y1'])

    for t in textos:
        try:
            x = float(t['x'])
            y = float(t['y'])
            texto = str(t['texto']).strip().upper()
            if not texto:
                continue

            match = REGEX_PINOS_FUNCIONAIS.search(texto)
            if not match:
                continue

            funcao_original = str(match.group(0)).upper()
            funcao_canonica = str(normalizar_funcao(funcao_original))

            dist_topo = abs(y - chip_y0)
            dist_base = abs(y - chip_y1)
            dist_esq = abs(x - chip_x0)
            dist_dir = abs(x - chip_x1)
            menor = min(dist_topo, dist_base, dist_esq, dist_dir)
            if menor > dist_max:
                continue

            if menor == dist_topo:
                lado = 'TOPO'
                pos_rel = (x - chip_x0) / max(chip_x1 - chip_x0, 1)
                pino_est = int(pos_rel * 30) + 1
            elif menor == dist_base:
                lado = 'BASE'
                pos_rel = (x - chip_x0) / max(chip_x1 - chip_x0, 1)
                pino_est = int((1 - pos_rel) * 30) + 31
            elif menor == dist_esq:
                lado = 'ESQUERDA'
                pos_rel = (y - chip_y0) / max(chip_y1 - chip_y0, 1)
                pino_est = int(pos_rel * 20) + 61
            else:
                lado = 'DIREITA'
                pos_rel = (y - chip_y0) / max(chip_y1 - chip_y0, 1)
                pino_est = int((1 - pos_rel) * 20) + 81

            cor = str(obter_cor_fio(funcao_canonica, ferramenta_cores))
            confianca = int(t.get('confianca', 70))

            pinos_encontrados.append({
                'pino': int(pino_est),
                'lado': str(lado),
                'funcao': str(funcao_canonica),
                'texto_original': str(texto),
                'cor_fio': str(cor),
                'ferramenta': str(ferramenta),
                'confianca': int(confianca)
            })
        except:
            continue

    vistos = set()
    unicos = []
    for p in sorted(pinos_encontrados, key=lambda x: (x['lado'], x['pino'])):
        chave = (p['funcao'], p['lado'])
        if chave not in vistos:
            vistos.add(chave)
            unicos.append(p)

    return to_native(unicos)


@monitorar(modulo='MPU')
def processar_modo_mpu(dados_pagina, ferramenta='CarProg_A10'):
    if not dados_pagina:
        return {'modo': 'erro', 'mensagem': 'dados_pagina é None', 'pinos': [], 'num_pinos': 0}

    try:
        modo_mpu, chip_rect = detectar_modo_mpu(dados_pagina)
    except Exception as e:
        logger.error(f"Erro na detecção MPU: {e}", extra={'modulo': 'MPU'})
        return {'modo': 'erro', 'mensagem': str(e), 'pinos': [], 'num_pinos': 0}

    if not modo_mpu or not chip_rect:
        return {'modo': 'normal', 'mensagem': 'Não detectado como MPU', 'pinos': [], 'num_pinos': 0}

    try:
        pinos = extrair_pinos_mpu(dados_pagina, chip_rect, ferramenta)
    except Exception as e:
        logger.error(f"Erro na extração de pinos: {e}", extra={'modulo': 'MPU'})
        return {'modo': 'mpu', 'chip': to_native(chip_rect), 'pinos': [], 'num_pinos': 0, 'ferramenta': str(ferramenta)}

    return to_native({
        'modo': 'mpu',
        'chip': to_native(chip_rect),
        'pinos': pinos,
        'num_pinos': int(len(pinos)),
        'ferramenta': str(ferramenta)
    })