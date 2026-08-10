"""
Módulo 3 - Restauração de Imagens Degradadas.
Recebe uma imagem (BGR) e retorna sua versão binarizada e esqueletizada,
corrigindo iluminação, ruído e tracejados.
"""

import cv2
import numpy as np
import time
from logger_erros import logger, monitorar, ErroExtracao, Severidade
from skimage.morphology import skeletonize

def upsample_se_necessario(imagem, dpi_alvo=300, usar_ia=False):
    h, w = imagem.shape[:2]
    if max(h, w) < 2000:
        fator = min(4.0, 2000 / max(h, w))
        imagem = cv2.resize(imagem, None, fx=fator, fy=fator, interpolation=cv2.INTER_LANCZOS4)
    return imagem

def equalizar_iluminacao(imagem):
    try:
        gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) if len(imagem.shape) == 3 else imagem
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    except Exception as e:
        logger.error(f"Erro ao equalizar iluminação: {e}", extra={'modulo': 'M3'})
        return imagem

def reduzir_ruido(imagem):
    try:
        return cv2.GaussianBlur(imagem, (3, 3), 0)
    except Exception as e:
        logger.error(f"Erro ao reduzir ruído: {e}", extra={'modulo': 'M3'})
        return imagem

def binarizar_adaptativo(imagem):
    try:
        gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) if len(imagem.shape) == 3 else imagem
        _, binaria = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        return binaria
    except Exception as e:
        logger.error(f"Erro na binarização: {e}", extra={'modulo': 'M3'})
        return imagem

def fechamento_direcional(binaria):
    kernel_h = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 7))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 1))
    fechada = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel_h)
    fechada = cv2.morphologyEx(fechada, cv2.MORPH_CLOSE, kernel_v)
    return fechada

def esqueletizar(binaria):
    """Usa skimage.skeletonize (rápido e estável)."""
    if binaria.dtype != np.uint8:
        binaria = binaria.astype(np.uint8)
    _, binaria = cv2.threshold(binaria, 127, 255, cv2.THRESH_BINARY)
    # skimage espera boolean, True = foreground
    skeleton_bool = skeletonize(binaria > 0)
    skel = (skeleton_bool * 255).astype(np.uint8)
    return skel

@monitorar(modulo='M3')
def restaurar_imagem(imagem_bgr, dpi_alvo=300, upscale_ia=False):
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

    logger.info(">>> Iniciando esqueletização (skimage)...", extra={'modulo': 'M3'})
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
