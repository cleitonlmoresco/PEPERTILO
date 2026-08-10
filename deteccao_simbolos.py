import cv2
import numpy as np
import logging
import os
import pytesseract

# Configuração automática do caminho do Tesseract para Windows
if os.name == 'nt':
    caminho_tesseract = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(caminho_tesseract):
        pytesseract.pytesseract.tesseract_cmd = caminho_tesseract

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

logger = logging.getLogger('pipeline')

class DetetorSimbolos:
    def __init__(self):
        self.ocr_paddle = None
        self.inicializar_ocr()

    def inicializar_ocr(self):
        if PaddleOCR is not None:
            try:
                self.ocr_paddle = PaddleOCR(use_angle_cls=True, lang='en')
                logger.info("PaddleOCR inicializado com sucesso.", extra={'modulo': 'ocr'})
            except Exception as e:
                logger.warning(f"Falha ao inicializar PaddleOCR: {e}. Usando Tesseract como fallback.", extra={'modulo': 'ocr'})
                self.ocr_paddle = None
        else:
            logger.warning("PaddleOCR não instalado. Usando Tesseract.", extra={'modulo': 'ocr'})

    def extrair_texto(self, imagem):
        textos = []
        
        # 1. Tenta extração com PaddleOCR
        if self.ocr_paddle is not None:
            try:
                resultado = self.ocr_paddle.ocr(imagem, cls=True)
                
                # Correção Lógica: Tratar o retorno [None] que o Paddle gera quando não acha texto
                if resultado and resultado[0] is not None:
                    for linha in resultado[0]:
                        box = linha[0]
                        texto = linha[1][0]
                        confianca = float(linha[1][1])
                        textos.append({
                            'texto': texto,
                            'confianca': confianca,
                            'box': box,
                            'origem': 'paddle'
                        })
                logger.info(f"PaddleOCR: {len(textos)} textos extraídos", extra={'modulo': 'ocr'})
                return textos  # Retorna o resultado mesmo se for vazio (confiamos no Paddle se não der erro)
            except Exception as e:
                logger.error(f"Erro no processamento PaddleOCR, acionando fallback: {e}", extra={'modulo': 'ocr'})

        # 2. Fallback para Tesseract (SÓ roda se o Paddle der erro ou não existir)
        try:
            gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) if len(imagem.shape) == 3 else imagem
            dados = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, lang='eng')
            
            n_boxes = len(dados['text'])
            for i in range(n_boxes):
                texto = dados['text'][i].strip()
                if texto:
                    conf = float(dados['conf'][i])
                    if conf > 0:
                        left, top, width, height = dados['left'][i], dados['top'][i], dados['width'][i], dados['height'][i]
                        box = [
                            [left, top],
                            [left + width, top],
                            [left + width, top + height],
                            [left, top + height]
                        ]
                        textos.append({
                            'texto': texto,
                            'confianca': conf / 100.0,
                            'box': box,
                            'origem': 'tesseract'
                        })
            
            logger.info(f"Tesseract (fallback): {len(textos)} textos extraídos", extra={'modulo': 'ocr'})
            return textos
        except Exception as e:
            logger.error(f"Erro no Tesseract fallback: {e}", extra={'modulo': 'ocr'})
            return []
