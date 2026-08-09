"""
Módulo 4 - Detecção de Símbolos em Imagens Restauradas.
Recebe imagem binarizada/esqueletizada do Módulo 3 e identifica
componentes, pinos, textos e emendas com monitoramento de desempenho.
"""

import cv2
import numpy as np
import re
import time
import traceback
import os
import warnings
from logger_erros import logger, monitorar, ErroExtracao, ErroPipeline, Severidade

# Suprimir warnings de compatibilidade
warnings.filterwarnings('ignore')
os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_enable_pir_in_executor'] = '0'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

CONFIG = {
    'area_min_componente': 500,
    'area_min_emenda': 10,
    'area_max_emenda': 200,
    'circularidade_min': 0.7,
    'dist_pino_borda': 15,
    'tam_min_fonte': 6,
}

RE_PINO = re.compile(r'^[A-Z]\d{1,2}$')


def to_native(obj, depth=0):
    """Converte QUALQUER objeto numpy para tipo Python nativo."""
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


def extrair_retangulos(binaria, canvas):
    """Encontra contornos fechados que representam componentes retangulares."""
    resultado = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(resultado) == 2:
        contours, _ = resultado
    else:
        _, contours, _ = resultado

    retangulos = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < CONFIG['area_min_componente']:
            continue
        area_contorno = cv2.contourArea(cnt)
        if area_contorno < area * 0.3:
            continue
        cx, cy = x + w/2, y + h/2
        if not (canvas[0] <= cx <= canvas[2] and canvas[1] <= cy <= canvas[3]):
            continue
        retangulos.append({
            'x0': float(x), 'y0': float(y),
            'x1': float(x+w), 'y1': float(y+h),
            'width': float(w), 'height': float(h),
            'area': float(area)
        })
    return to_native(retangulos)


def extrair_textos_ocr(imagem):
    """Usa PaddleOCR ou Tesseract para extrair textos da imagem."""
    textos = []
    
    # Tentar PaddleOCR primeiro
    try:
        from paddleocr import PaddleOCR
        logger.info("Inicializando PaddleOCR...", extra={'modulo': 'M4'})
        
        # Desabilitar PIR internamente
        os.environ['FLAGS_enable_pir_api'] = '1'
        os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
        
        ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
        
        if len(imagem.shape) == 2:
            img_rgb = cv2.cvtColor(imagem, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        
        resultado = ocr.ocr(img_rgb, cls=False)
        
        if resultado and resultado[0]:
            for line in resultado[0]:
                try:
                    box = line[0]
                    texto = line[1][0]
                    conf = line[1][1]
                    
                    if conf < 0.3:  # Filtrar baixa confiança
                        continue
                    
                    xc = (box[0][0] + box[2][0]) / 2
                    yc = (box[0][1] + box[2][1]) / 2
                    altura = abs(box[2][1] - box[0][1])
                    
                    if altura < CONFIG['tam_min_fonte']:
                        continue
                    
                    textos.append({
                        'x': float(xc), 'y': float(yc),
                        'texto': str(texto),
                        'tam': float(altura),
                        'confianca': float(conf)
                    })
                except Exception as e:
                    logger.debug(f"Erro ao processar linha OCR: {e}", extra={'modulo': 'M4'})
                    continue
        
        if textos:
            logger.info(f"PaddleOCR: {len(textos)} textos extraídos", extra={'modulo': 'M4'})
            return to_native(textos)
            
    except Exception as e:
        logger.warning(f"PaddleOCR falhou ({type(e).__name__}): {str(e)[:100]}", extra={'modulo': 'M4'})
    
    # Fallback: Tesseract
    try:
        logger.info("Tentando Tesseract como fallback...", extra={'modulo': 'M4'})
        import pytesseract
        
        if len(imagem.shape) == 3:
            gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        else:
            gray = imagem
        
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        
        for i in range(len(data['text'])):
            txt = data['text'][i].strip()
            if not txt or int(data['conf'][i]) < 30:
                continue
            
            x = float(data['left'][i] + data['width'][i] / 2)
            y = float(data['top'][i] + data['height'][i] / 2)
            
            textos.append({
                'x': x, 'y': y,
                'texto': txt,
                'tam': float(data['height'][i]),
                'confianca': int(data['conf'][i])
            })
        
        if textos:
            logger.info(f"Tesseract: {len(textos)} textos extraídos", extra={'modulo': 'M4'})
            return to_native(textos)
            
    except Exception as e:
        logger.debug(f"Tesseract também falhou: {str(e)[:100]}", extra={'modulo': 'M4'})
    
    logger.warning("Nenhum OCR disponível - retornando lista vazia", extra={'modulo': 'M4'})
    return []


def extrair_emendas(binaria):
    """Detecta pequenos círculos que representam emendas (soldas)."""
    resultado = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(resultado) == 2:
        contours, _ = resultado
    else:
        _, contours, _ = resultado

    emendas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < CONFIG['area_min_emenda'] or area > CONFIG['area_max_emenda']:
            continue
        perimetro = cv2.arcLength(cnt, True)
        if perimetro == 0:
            continue
        circularidade = 4 * np.pi * area / (perimetro * perimetro)
        if circularidade < CONFIG['circularidade_min']:
            continue
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            emendas.append((float(cx), float(cy)))
    return to_native(emendas)


def extrair_linhas_do_esqueleto(esqueleto):
    """Converte imagem esqueletizada em segmentos de linha."""
    linhas = []
    if esqueleto is None:
        return linhas

    try:
        esqueleto = np.asarray(esqueleto, dtype=np.uint8)
        if esqueleto.ndim == 3:
            esqueleto = cv2.cvtColor(esqueleto, cv2.COLOR_BGR2GRAY)
        if esqueleto.size == 0:
            return linhas

        _, esqueleto = cv2.threshold(esqueleto, 127, 255, cv2.THRESH_BINARY)
        lines = cv2.HoughLinesP(esqueleto, 1, np.pi/180, threshold=30, minLineLength=10, maxLineGap=5)

        if lines is None:
            return linhas

        for line in lines:
            try:
                flat = np.asarray(line, dtype=np.float64).flatten()
                if len(flat) >= 4:
                    x1, y1, x2, y2 = float(flat[0]), float(flat[1]), float(flat[2]), float(flat[3])
                    linhas.append(((x1, y1), (x2, y2)))
            except:
                continue

    except Exception as e:
        logger.error(f"extrair_linhas_do_esqueleto falhou: {e}", extra={'modulo': 'M4'})
        return []

    return to_native(linhas)


@monitorar(modulo='M4')
def detectar_simbolos(imagem_binaria, imagem_original=None):
    """Detecta todos os símbolos na imagem binária restaurada com métricas de tempo."""
    if imagem_binaria is None:
        raise ErroExtracao("Imagem binária é None", severidade=Severidade.ALTA)

    h, w = imagem_binaria.shape[:2]
    canvas = (0, 0, w, h)

    logger.info(f">>> Iniciando detecção de símbolos na área {w}x{h}", extra={'modulo': 'M4'})
    
    t0 = time.perf_counter()
    retangulos = extrair_retangulos(imagem_binaria, canvas)
    t1 = time.perf_counter()
    logger.debug(f"[Métrica] extrair_retangulos levou {t1-t0:.2f}s", extra={'modulo': 'M4'})

    logger.info(">>> Iniciando OCR (processo pesado)...", extra={'modulo': 'M4'})
    textos = extrair_textos_ocr(imagem_original if imagem_original is not None else imagem_binaria)
    t2 = time.perf_counter()
    logger.info(f">>> [Métrica Crítica] extrair_textos_ocr levou {t2-t1:.2f}s ({len(textos)} textos)", extra={'modulo': 'M4'})

    curvas = extrair_emendas(imagem_binaria)
    t3 = time.perf_counter()
    logger.debug(f"[Métrica] extrair_emendas levou {t3-t2:.2f}s", extra={'modulo': 'M4'})

    logger.info(f"Símbolos detectados: {len(retangulos)} retângulos, {len(textos)} textos, {len(curvas)} emendas",
                extra={'modulo': 'M4'})

    resultado = {
        'linhas': [],
        'textos': textos,
        'retangulos': retangulos,
        'curvas': curvas,
        'canvas': canvas,
        'width': w,
        'height': h
    }
    return to_native(resultado)


@monitorar(modulo='M4')
def processar_imagem_restaurada(esqueleto, binaria, original=None):
    """Processa a imagem restaurada e retorna estrutura blindada contra falhas."""
    logger.info(">>> Iniciando Módulo 4: Processamento de Imagem Restaurada...", extra={'modulo': 'M4'})
    t_inicio = time.perf_counter()
    
    try:
        # Extração de Símbolos e Textos
        simbolos = detectar_simbolos(binaria, original)
        
        # Extração de Linhas
        t_linhas_inicio = time.perf_counter()
        logger.info(">>> Iniciando extração de linhas (HoughLinesP)...", extra={'modulo': 'M4'})
        simbolos['linhas'] = extrair_linhas_do_esqueleto(esqueleto)
        t_linhas_fim = time.perf_counter()
        logger.debug(f"[Métrica] extrair_linhas_do_esqueleto levou {t_linhas_fim - t_linhas_inicio:.2f}s", extra={'modulo': 'M4'})
        
        # Sanitização e Finalização
        resultado_final = to_native(simbolos)
        
        t_total = time.perf_counter() - t_inicio
        logger.info(f">>> Módulo 4 concluído com sucesso em {t_total:.2f}s. "
                    f"Total de Linhas vetorizadas: {len(resultado_final.get('linhas', []))}", extra={'modulo': 'M4'})
        
        return resultado_final
        
    except Exception as e:
        logger.error(f"Erro fatal durante extração no Módulo 4: {e}\n{traceback.format_exc()}", extra={'modulo': 'M4'})
        raise ErroPipeline(
            mensagem="Falha durante a extração de OCR ou Símbolos",
            modulo='M4',
            severidade=Severidade.CRITICA,
            causa=e
        )


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Uso: python deteccao_simbolos.py <imagem_restaurada.png>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Erro ao carregar imagem.")
        sys.exit(1)

    _, binaria = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    esqueleto = binaria

    resultado = processar_imagem_restaurada(esqueleto, binaria, original=cv2.imread(sys.argv[1]))
    print(f"Linhas: {len(resultado['linhas'])}")
    print(f"Textos: {len(resultado['textos'])}")
    print(f"Retângulos: {len(resultado['retangulos'])}")
    print(f"Emendas: {len(resultado['curvas'])}")
