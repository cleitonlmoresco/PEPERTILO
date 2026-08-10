"""
Roteador - Módulo Central de Encaminhamento.
Classifica o arquivo e encaminha para os módulos de processamento adequados.
Suporte a diagramas elétricos, manuais de programadores e datasheets.
"""

import os
import sys
import traceback
import cv2
import numpy as np
from pathlib import Path
from classificador import classificar_arquivo, TIPOS_VALIDOS
from logger_erros import logger, monitorar, ErroPipeline, Severidade, ColetorErros
from utils import to_native

from extracao_vetorial import extrair_primitivas_vetorial
from grafo_rastreador import processar_diagrama_multipagina, processar_diagrama
from restauracao_img import restaurar_imagem
from deteccao_simbolos import processar_imagem_restaurada
from modo_mpu import processar_modo_mpu
from extracao_datasheet import extrair_datasheet
# M11 - específico para CarProg (SEM OCR)
try:
    from extracao_carprog import extrair_carprog
    M11_DISPONIVEL = True
except ImportError:
    M11_DISPONIVEL = False
    logger.warning("M11 (extracao_carprog) não disponível", extra={'modulo': 'Roteador'})
# M9 - fallback simples com Tesseract
try:
    from extracao_manual import extrair_manual_com_tesseract
    M9_DISPONIVEL = True
except ImportError:
    M9_DISPONIVEL = False
    logger.warning("M9 não disponível", extra={'modulo': 'Roteador'})

# M10 - extração especializada para programadores
try:
    from extracao_programadores import extrair_programador_com_tesseract
    M10_DISPONIVEL = True
    logger.info("M10 disponível", extra={'modulo': 'Roteador'})
except ImportError as e:
    M10_DISPONIVEL = False
    logger.warning(f"M10 não disponível: {e}", extra={'modulo': 'Roteador'})

DEBUG_MODE = os.environ.get('DEBUG_MODE', 'false').lower() == 'true'
DEBUG_DIR = Path('debug')
if DEBUG_MODE:
    DEBUG_DIR.mkdir(exist_ok=True)


# ============================================================
# PROCESSAR PDF RASTERIZADO
# ============================================================
def processar_pdf_rasterizado(caminho_pdf, limite_paginas=0):
    logger.info(">>> INICIANDO PROCESSAMENTO DE PDF RASTERIZADO...", extra={'modulo': 'Roteador'})
    coletor = ColetorErros()

    try:
        import fitz
        doc = fitz.open(caminho_pdf)
    except Exception as e:
        return {'status': 'erro', 'mensagem': f'Não foi possível abrir o PDF: {str(e)}'}

    total_paginas = len(doc)
    if total_paginas == 0:
        doc.close()
        return {'status': 'erro', 'mensagem': 'PDF não contém páginas.'}

    dados_por_pagina = {}
    pinos_mpu = []
    paginas_processadas = 0
    total_textos = 0
    total_retangulos = 0

    for i, page in enumerate(doc):
        if limite_paginas > 0 and i >= limite_paginas:
            logger.info(f"Limite de {limite_paginas} páginas atingido.", extra={'modulo': 'Roteador'})
            break

        try:
            logger.info(f">>> Extraindo página {i+1}/{total_paginas}", extra={'modulo': 'Roteador'})

            pix = page.get_pixmap(dpi=300)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if DEBUG_MODE:
                cv2.imwrite(str(DEBUG_DIR / f"pagina_{i+1}_original.png"), img)

            esqueleto, binaria = restaurar_imagem(img)
            dados_pagina = processar_imagem_restaurada(esqueleto, binaria, original=img)

            linhas = dados_pagina.get('linhas', [])
            textos = dados_pagina.get('textos', [])
            retangulos = dados_pagina.get('retangulos', [])
            emendas = dados_pagina.get('curvas', [])

            dados_por_pagina[i+1] = {
                'linhas': linhas,
                'textos': textos,
                'retangulos': retangulos,
                'curvas': emendas,
                'canvas': dados_pagina.get('canvas', (0, 0, 1000, 1000)),
                'width': dados_pagina.get('width', 1000),
                'height': dados_pagina.get('height', 1000)
            }

            total_textos += len(textos)
            total_retangulos += len(retangulos)
            paginas_processadas += 1

            resultado_mpu = processar_modo_mpu(dados_pagina, ferramenta='CarProg_A10')
            if resultado_mpu.get('modo') == 'mpu':
                logger.info(">>> MODO MPU ATIVADO <<<", extra={'modulo': 'MPU'})
                pinos_mpu.extend(resultado_mpu.get('pinos', []))

            del img, esqueleto, binaria, dados_pagina

        except Exception as e:
            coletor.adicionar_erro(
                ErroPipeline(f"Erro na página {i+1}: {str(e)}", modulo='Roteador', causa=e),
                severidade=Severidade.MEDIA
            )
            continue

    doc.close()

    if paginas_processadas == 0:
        return {'status': 'erro', 'mensagem': 'Nenhuma página processada com sucesso'}

    if pinos_mpu:
        return {
            'status': 'ok',
            'modulo': 'MPU',
            'modo': 'mpu',
            'pinos': to_native(pinos_mpu),
            'num_pinos': len(pinos_mpu),
            'num_paginas': paginas_processadas,
            'ferramenta': 'CarProg_A10'
        }

      # ============================================================
    # DETECÇÃO DE MANUAL - ORDEM: M11 → M10 → M6 → grafo
    # ============================================================
    if total_retangulos < 30 and total_textos > total_retangulos * 3:
        logger.warning(
            f"Detectado possível manual: {total_textos} textos, {total_retangulos} retângulos. "
            f"Ativando extração especializada.",
            extra={'modulo': 'Roteador'}
        )

        # ---- TENTATIVA 0: M11 (específico CarProg, SEM OCR) ----
        nome_arquivo = os.path.basename(caminho_pdf)
        if M11_DISPONIVEL and ("CarProg" in nome_arquivo or "HC12" in nome_arquivo or "9S12" in nome_arquivo):
            try:
                logger.info(">>> Tentando M11 (extração específica CarProg, sem OCR)...", extra={'modulo': 'Roteador'})
                pin_func = extrair_carprog(caminho_pdf, limite_paginas)
                if pin_func:
                    logger.info(f"M11 extraiu {len(pin_func)} funções", extra={'modulo': 'Roteador'})
                    return {
                        'status': 'ok',
                        'modulo': 'M11',
                        'funcoes': to_native(pin_func),
                        'num_pinos': len(pin_func),
                        'num_paginas': paginas_processadas,
                        'mensagem': 'CarProg processado com M11 (regex)'
                    }
                else:
                    logger.warning("M11 não extraiu funções.", extra={'modulo': 'Roteador'})
            except Exception as e:
                logger.error(f"M11 falhou: {e}", extra={'modulo': 'Roteador'})

        # ---- TENTATIVA 1: M10 ----
        if M10_DISPONIVEL:
            try:
                logger.info(">>> Tentando M10 (extração avançada para programadores)...", extra={'modulo': 'Roteador'})
                pin_func = extrair_programador_com_tesseract(caminho_pdf, limite_paginas)
                if pin_func:
                    logger.info(f"M10 extraiu {len(pin_func)} funções", extra={'modulo': 'Roteador'})
                    return {
                        'status': 'ok',
                        'modulo': 'M10',
                        'funcoes': to_native(pin_func),
                        'num_pinos': len(pin_func),
                        'num_paginas': paginas_processadas,
                        'mensagem': 'Manual processado com M10 (layout+OCR)'
                    }
                else:
                    logger.warning("M10 não extraiu funções.", extra={'modulo': 'Roteador'})
            except Exception as e:
                logger.error(f"M10 falhou: {e}", extra={'modulo': 'Roteador'})
        else:
            logger.warning("M10 não disponível.", extra={'modulo': 'Roteador'})

        # ---- TENTATIVA 2: M6 (com fallback M9) ----
        try:
            logger.info(">>> Tentando M6 (datasheet genérico)...", extra={'modulo': 'Roteador'})
            pin_func = extrair_datasheet(caminho_pdf, limite_paginas)
            if pin_func:
                logger.info(f"M6 extraiu {len(pin_func)} funções", extra={'modulo': 'Roteador'})
                return {
                    'status': 'ok',
                    'modulo': 'M6',
                    'funcoes': to_native(pin_func),
                    'num_pinos': len(pin_func),
                    'num_paginas': paginas_processadas,
                    'mensagem': 'Manual processado com M6/M9'
                }
            else:
                logger.warning("M6 não extraiu funções.", extra={'modulo': 'Roteador'})
        except Exception as e:
            logger.error(f"M6 falhou: {e}", extra={'modulo': 'Roteador'})

        # ---- SE NADA FUNCIONOU ----
        return {
            'status': 'erro',
            'mensagem': 'Arquivo parece ser um manual, mas não foi possível extrair a tabela de pinos.',
            'num_paginas': paginas_processadas
        }
    # ============================================================
    # SE NÃO É MANUAL, TENTA CONSTRUIR GRAFO (DIAGRAMA)
    # ============================================================
    try:
        conexoes, G, pinos, perifs = processar_diagrama_multipagina(dados_por_pagina)
        return {
            'status': 'ok',
            'modulo': 'M5',
            'conexoes': to_native(conexoes),
            'num_conexoes': len(conexoes),
            'num_pinos': len(pinos),
            'num_paginas': paginas_processadas
        }
    except Exception as e:
        logger.error(f"Erro ao construir grafo: {e}", extra={'modulo': 'Roteador'})
        return {
            'status': 'erro',
            'mensagem': f'Falha ao construir grafo: {str(e)}',
            'num_paginas': paginas_processadas
        }

# ============================================================
# DEMAIS FUNÇÕES (sem alterações)
# ============================================================

def processar_pdf_vetorial(caminho):
    logger.info(f"Processando PDF vetorial: {caminho}", extra={'modulo': 'Roteador'})
    try:
        dados = extrair_primitivas_vetorial(caminho)
        if not dados:
            return {'status': 'erro', 'mensagem': 'Nenhum dado extraído do PDF.'}
        dados = to_native(dados)
        if len(dados) > 1:
            conexoes, G, pinos, perifs = processar_diagrama_multipagina(dados)
        else:
            conexoes, G, pinos, perifs = processar_diagrama(dados[1])
        return {
            'status': 'ok',
            'modulo': 'M5',
            'arquivo': str(caminho),
            'conexoes': to_native(conexoes),
            'num_conexoes': int(len(conexoes)),
            'num_pinos_ecu': int(len(pinos)),
            'num_paginas': int(len(dados))
        }
    except Exception as e:
        logger.error(f"Erro no processamento vetorial: {e}\n{traceback.format_exc()}", extra={'modulo': 'Roteador'})
        return {'status': 'erro', 'mensagem': str(e)}

def processar_imagem_limpa(caminho_imagem):
    logger.info(f"Processando imagem limpa: {caminho_imagem}", extra={'modulo': 'Roteador'})
    try:
        img = cv2.imread(caminho_imagem)
        if img is None:
            return {'status': 'erro', 'mensagem': 'Não foi possível abrir a imagem.'}
        esqueleto, binaria = restaurar_imagem(img)
        dados_pagina = processar_imagem_restaurada(esqueleto, binaria, original=img)
        dados_pagina = to_native(dados_pagina)
        conexoes, G, pinos, perifs = processar_diagrama(dados_pagina)
        return {
            'status': 'ok',
            'modulo': 'M5',
            'conexoes': to_native(conexoes),
            'num_conexoes': int(len(conexoes)),
            'num_pinos': int(len(pinos))
        }
    except Exception as e:
        logger.error(f"Erro no processamento de imagem: {e}\n{traceback.format_exc()}", extra={'modulo': 'Roteador'})
        return {'status': 'erro', 'mensagem': str(e)}

def processar_foto_celular(caminho_imagem):
    return processar_imagem_limpa(caminho_imagem)

def processar_manual(caminho_pdf):
    logger.info(f"Processando manual explicitamente: {caminho_pdf}", extra={'modulo': 'Roteador'})
    try:
        pin_func = extrair_datasheet(caminho_pdf)
        if pin_func:
            return {
                'status': 'ok',
                'modulo': 'M6',
                'funcoes': to_native(pin_func),
                'num_pinos': len(pin_func),
                'mensagem': 'Datasheet extraído com sucesso'
            }
        else:
            return {'status': 'erro', 'mensagem': 'Nenhuma tabela de pinos encontrada.'}
    except Exception as e:
        logger.error(f"Erro no processamento manual: {e}", extra={'modulo': 'Roteador'})
        return {'status': 'erro', 'mensagem': str(e)}

def processar_desconhecido(caminho):
    logger.warning(f"Tipo desconhecido: {caminho}", extra={'modulo': 'Roteador'})
    return {'status': 'erro', 'mensagem': 'Tipo de arquivo não suportado.'}

ROTAS = {
    'pdf_vetorial': processar_pdf_vetorial,
    'pdf_rasterizado': processar_pdf_rasterizado,
    'imagem_limpa': processar_imagem_limpa,
    'foto_celular': processar_foto_celular,
    'manual': processar_manual,
    'desconhecido': processar_desconhecido,
}

@monitorar(modulo='Roteador')
def rotear_arquivo(caminho_arquivo, limite_paginas=0):
    if not os.path.exists(caminho_arquivo):
        raise ErroPipeline(f"Arquivo não encontrado: {caminho_arquivo}", modulo='Roteador', severidade=Severidade.CRITICA)

    tipo = classificar_arquivo(caminho_arquivo)
    descricao = TIPOS_VALIDOS.get(tipo, 'Desconhecido')

    logger.info(f"Arquivo classificado como: {tipo} ({descricao})", extra={'modulo': 'Roteador'})

    funcao = ROTAS.get(tipo, processar_desconhecido)
    if tipo in ('pdf_rasterizado', 'manual'):
        resultado = funcao(caminho_arquivo, limite_paginas)
    else:
        resultado = funcao(caminho_arquivo)

    return {
        'arquivo': str(caminho_arquivo),
        'tipo': str(tipo),
        'descricao': str(descricao),
        'resultado': to_native(resultado)
    }

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Uso: python roteador.py <arquivo> [limite_paginas]")
        sys.exit(1)

    limite = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    try:
        resultado = rotear_arquivo(sys.argv[1], limite)
        print(f"\nTipo: {resultado['tipo']} - {resultado['descricao']}")
        if resultado['resultado']['status'] == 'ok':
            if resultado['resultado'].get('modo') == 'mpu':
                print(f"Pinos MPU encontrados: {resultado['resultado']['num_pinos']}")
            elif resultado['resultado'].get('modulo') in ('M6', 'M9', 'M10'):
                print(f"Funções de pinos extraídas: {resultado['resultado']['num_pinos']}")
            else:
                print(f"Conexões encontradas: {resultado['resultado'].get('num_conexoes', 0)}")
        else:
            print(f"Erro: {resultado['resultado'].get('mensagem', 'Desconhecido')}")
    except ErroPipeline as e:
        logger.critical(f"Falha crítica: {e.to_dict()}")
        sys.exit(1)
