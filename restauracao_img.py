import cv2
import numpy as np
import logging

logger = logging.getLogger('pipeline')

def equalizar_iluminacao(imagem):
    """
    Correção Lógica: Substituído o cálculo de absdiff por CLAHE.
    Isso mantém o fundo branco e as linhas/textos do esquema elétrico pretos, 
    ideal para a leitura do OCR.
    """
    try:
        gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) if len(imagem.shape) == 3 else imagem
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    except Exception as e:
        logger.error(f"Erro ao equalizar iluminação: {e}", extra={'modulo': 'restauracao'})
        return imagem

def reduzir_ruido(imagem):
    """
    Correção Lógica: Utilização de GaussianBlur para preservação de bordas de fontes
    em diagramas, substituindo o algoritmo mais pesado (NLMeans).
    """
    try:
        return cv2.GaussianBlur(imagem, (3, 3), 0)
    except Exception as e:
        logger.error(f"Erro ao reduzir ruído: {e}", extra={'modulo': 'restauracao'})
        return imagem

def binarizar_adaptativo(imagem):
    """
    Binarização Otsu clássica.
    Como aplicamos CLAHE antes, a imagem não está com as cores invertidas.
    """
    try:
        gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) if len(imagem.shape) == 3 else imagem
        _, binaria = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return binaria
    except Exception as e:
        logger.error(f"Erro na binarização Otsu: {e}", extra={'modulo': 'restauracao'})
        return imagem

def restaurar_imagem_completa(imagem):
    try:
        eq = equalizar_iluminacao(imagem)
        ruido = reduzir_ruido(eq)
        binaria = binarizar_adaptativo(ruido)
        return binaria
    except Exception as e:
        logger.error(f"Erro no pipeline de restauração: {e}", extra={'modulo': 'restauracao'})
        return imagem
