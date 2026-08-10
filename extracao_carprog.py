"""
Módulo 11 - Extração específica para manuais CarProg (HC12, 9S12, etc.)
SEM OCR. Usa extração de texto direta do PDF + regex.
"""

import re
import fitz
from logger_erros import logger

# Padrões para extração
PADRAO_FIOS = re.compile(
    r'(BLUE|BROWN|GREY|RED|VIOLET|WHITE|GREEN|ORANGE|YELLOW|BLACK|PURPLE|PINK)\s*[-–]\s*([^\n;]+)',
    re.IGNORECASE
)

PADRAO_MCU = re.compile(
    r'(MC68HC912(B32|BC32|BD32|D60))\s+([A-Z0-9]+)',
    re.IGNORECASE
)

PADRAO_MASK = re.compile(
    r'(4C11W|F68K|4F73K|0K75F|0K13J|L26M|9H91F|J15G|K29E)',
    re.IGNORECASE
)

PADRAO_PINO_FUNCAO = re.compile(
    r'([A-Z0-9]{2,5})\s+[-–]\s+([A-Za-z0-9\s]+)',
    re.IGNORECASE
)

# Mapeamento manual para termos que frequentemente são extraídos incorretamente
MAPA_CORRECAO = {
    'BKGD': 'BKGD (Background Debug)',
    'RESET': 'Reset signal',
    'GND': 'Ground',
    'VDD': 'Positive supply',
    'VSS': 'Ground supply',
}

def extrair_carprog(caminho_pdf, limite_paginas=0):
    """
    Extrai dados de manuais CarProg usando texto direto do PDF.
    Retorna dict {pino: funcao}.
    """
    logger.info(f"[M11] Iniciando extração CarProg: {caminho_pdf}", extra={'modulo': 'M11'})

    try:
        doc = fitz.open(caminho_pdf)
    except Exception as e:
        logger.error(f"[M11] Erro ao abrir PDF: {e}", extra={'modulo': 'M11'})
        return {}

    total = len(doc)
    if limite_paginas > 0:
        total = min(total, limite_paginas)

    texto_completo = ""
    for i in range(total):
        page = doc[i]
        texto = page.get_text()
        texto_completo += texto + "\n"
        logger.debug(f"[M11] Página {i+1}: {len(texto)} caracteres", extra={'modulo': 'M11'})

    doc.close()

    if not texto_completo.strip():
        logger.warning("[M11] Nenhum texto extraído do PDF. O PDF é uma imagem (não tem texto selecionável).", extra={'modulo': 'M11'})
        return {}

    resultado = {}

    # 1. Extrair fios coloridos (BLUE - Reset, etc.)
    for match in PADRAO_FIOS.finditer(texto_completo):
        cor = match.group(1).strip().upper()
        descricao = match.group(2).strip()
        descricao = re.sub(r'[;,]', '', descricao)
        if cor and descricao:
            resultado[cor] = descricao
            logger.debug(f"[M11] Fio: {cor} -> {descricao}", extra={'modulo': 'M11'})

    # 2. Extrair modelos MCU
    for match in PADRAO_MCU.finditer(texto_completo):
        modelo = match.group(1).strip()
        mask = match.group(3).strip()
        if modelo and mask:
            resultado[modelo] = f"Mask {mask}"
            logger.debug(f"[M11] MCU: {modelo} -> {mask}", extra={'modulo': 'M11'})

    # 3. Extrair máscaras (se não estiverem associadas a MCU)
    for match in PADRAO_MASK.finditer(texto_completo):
        mask = match.group(1).strip()
        if mask and mask not in str(resultado.values()):
            # Tenta encontrar contexto: linha anterior
            linhas = texto_completo.split('\n')
            for idx, linha in enumerate(linhas):
                if mask in linha:
                    partes = linha.split(mask)
                    antes = partes[0].strip()
                    if antes and len(antes) < 30 and any(c.isalpha() for c in antes):
                        resultado[antes] = f"Mask {mask}"
                        break
                    else:
                        resultado[mask] = "Máscara MPU"
                        break

    # 4. Se ainda não extraiu nada, tenta padrão genérico de pino-função
    if len(resultado) < 3:
        for match in PADRAO_PINO_FUNCAO.finditer(texto_completo):
            pino = match.group(1).strip().upper()
            funcao = match.group(2).strip()
            if pino and funcao and len(pino) >= 2 and len(funcao) >= 2:
                resultado[pino] = funcao

    # 5. Aplica correções manuais para termos conhecidos
    for k, v in resultado.items():
        if k in MAPA_CORRECAO:
            resultado[k] = MAPA_CORRECAO[k]

    # Filtra itens indesejados (palavras muito curtas ou óbvias)
    itens_filtrados = {}
    for k, v in resultado.items():
        if len(k) < 2 or k in ['TO', 'FOR', 'E', 'IF', 'DID', 'ALL', 'OR', 'AND', 'THE']:
            continue
        if len(v) < 2 or v.lower() in ['or', 'to', 'all', 'did', 'can']:
            continue
        # Remove chaves que são apenas números de página
        if k.isdigit() and len(k) == 1:
            continue
        itens_filtrados[k] = v

    logger.info(f"[M11] Extração concluída: {len(itens_filtrados)} pares", extra={'modulo': 'M11'})
    return itens_filtrados

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Uso: python extracao_carprog.py <arquivo.pdf>")
        sys.exit(1)
    resultado = extrair_carprog(sys.argv[1])
    for k, v in list(resultado.items())[:20]:
        print(f"{k} -> {v}")
