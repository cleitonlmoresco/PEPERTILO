"""
Módulo 3 - Restauração de Imagens Degradadas.
Recebe uma imagem (BGR) e retorna sua versão binarizada e esqueletizada,
corrigindo iluminação, ruído e tracejados.
"""

import cv2
import numpy as np
import time
from logger_erros import logger, monitorar, ErroExtracao, Severidade


def upsample_se_necessario(imagem, dpi_alvo=300, usar_ia=False):
    """Aumenta a resolução se a imagem for menor que o necessário."""
    h, w = imagem.shape[:2]
    if max(h, w) < 2000:
        fator = min(4.0, 2000 / max(h, w))
        imagem = cv2.resize(imagem, None, fx=fator, fy=fator, interpolation=cv2.INTER_LANCZOS4)
    return imagem


def equalizar_iluminacao(imagem):
    """Remove sombras e melhora contraste local usando CLAHE no canal L do Lab."""
    lab = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def reduzir_ruido(imagem):
    """Remove ruído sal-e-pimenta e suaviza texturas sem borrar bordas."""
    imagem = cv2.medianBlur(imagem, 3)
    imagem = cv2.bilateralFilter(imagem, d=9, sigmaColor=75, sigmaSpace=75)
    return imagem


def binarizar_adaptativo(imagem):
    """Binariza a imagem usando limiar adaptativo gaussiano."""
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) < 128:
        gray = cv2.bitwise_not(gray)
    binaria = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=51, C=2
    )
    return binaria


def fechamento_direcional(binaria):
    """Fecha buracos em linhas tracejadas sem unir fios paralelos."""
    kernel_h = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 15))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 1))
    fechada = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel_h)
    fechada = cv2.morphologyEx(fechada, cv2.MORPH_CLOSE, kernel_v)
    return fechada


def esqueletizar(binaria):
    """Aplica afinamento até obter esqueleto de 1 pixel de espessura."""
    # Garantir que é uint8 e binária
    if binaria.dtype != np.uint8:
        binaria = binaria.astype(np.uint8)
    _, binaria = cv2.threshold(binaria, 127, 255, cv2.THRESH_BINARY)
    
    # Tentar ximgproc
    try:
        if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning'):
            return cv2.ximgproc.thinning(binaria, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except:
        pass
    
    # Fallback manual
    skel = np.zeros(binaria.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = binaria.copy()
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel


@monitorar(modulo='M3')
def restaurar_imagem(imagem_bgr, dpi_alvo=300, upscale_ia=False):
    """
    Pipeline completo de restauração com rastreamento de gargalos de tempo.
    Retorna (esqueleto, binaria) como uint8.
    """
    if imagem_bgr is None:
        raise ErroExtracao("Imagem de entrada é None", severidade=Severidade.ALTA)

    logger.info(f">>> Iniciando restauração: shape={imagem_bgr.shape}", extra={'modulo': 'M3'})
    
    t0 = time.perf_counter()
    
    imagem = upsample_se_necessario(imagem_bgr, dpi_alvo, usar_ia=upscale_ia)
    t1 = time.perf_counter()
    logger.debug(f"[Métrica] upsample_se_necessario levou {t1-t0:.2f}s", extra={'modulo': 'M3'})

    imagem = equalizar_iluminacao(imagem)
    t2 = time.perf_counter()
    logger.debug(f"[Métrica] equalizar_iluminacao levou {t2-t1:.2f}s", extra={'modulo': 'M3'})

    imagem = reduzir_ruido(imagem)
    t3 = time.perf_counter()
    logger.debug(f"[Métrica] reduzir_ruido levou {t3-t2:.2f}s", extra={'modulo': 'M3'})

    binaria = binarizar_adaptativo(imagem)
    t4 = time.perf_counter()
    logger.debug(f"[Métrica] binarizar_adaptativo levou {t4-t3:.2f}s", extra={'modulo': 'M3'})

    binaria = fechamento_direcional(binaria)
    t5 = time.perf_counter()
    logger.debug(f"[Métrica] fechamento_direcional levou {t5-t4:.2f}s", extra={'modulo': 'M3'})

    # O principal suspeito de travamento
    logger.info(">>> Iniciando esqueletização (etapa pesada)...", extra={'modulo': 'M3'})
    esqueleto = esqueletizar(binaria)
    t6 = time.perf_counter()
    logger.info(f">>> [Métrica Crítica] esqueletizar levou {t6-t5:.2f}s", extra={'modulo': 'M3'})
    
    if esqueleto.dtype != np.uint8:
        esqueleto = esqueleto.astype(np.uint8)
    if binaria.dtype != np.uint8:
        binaria = binaria.astype(np.uint8)

    tempo_total = t6 - t0
    logger.info(f">>> Restauração concluída em {tempo_total:.2f}s", extra={'modulo': 'M3'})
    
    return esqueleto, binaria


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Uso: python restauracao_img.py <imagem>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print("Erro ao carregar imagem.")
        sys.exit(1)

    esq, binaria = restaurar_imagem(img)
    cv2.imwrite("restaurada_binaria.png", binaria)
    cv2.imwrite("restaurada_esqueleto.png", esq)
    print("Imagens salvas: restaurada_binaria.png, restaurada_esqueleto.png")
