"""
Roteador - Módulo Central de Encaminhamento.
Classifica o arquivo e encaminha para os módulos de processamento adequados.
Suporte a streaming de páginas para reduzir consumo de RAM.
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

# Módulos do pipeline
from extracao_vetorial import extrair_primitivas_vetorial
from grafo_rastreador import processar_diagrama_multipagina, processar_diagrama
from restauracao_img import restaurar_imagem
from deteccao_simbolos import processar_imagem_restaurada
from modo_mpu import processar_modo_mpu
from extracao_datasheet import extrair_datasheet

DEBUG_MODE = os.environ.get('DEBUG_MODE', 'false').lower() == 'true'
DEBUG_DIR = Path('debug')
if DEBUG_MODE:
    DEBUG_DIR.mkdir(exist_ok=True)

# ============================================================
# PROCESSAR PDF RASTERIZADO COM STREAMING E FALLBACK
# ============================================================
def processar_pdf_rasterizado(caminho_pdf, limite_paginas=0):
    """
    Processa PDF rasterizado com streaming: cada página é processada e seus dados
    são acumulados em estruturas leves (não armazena imagens).
    Se detectar que o PDF é um manual/datasheet (poucos elementos gráficos),
    aciona o M6 automaticamente.
    """
    logger.info(">>> INICIANDO PROCESSAMENTO STREAMING DO PDF...", extra={'modulo': 'Roteador'})
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

    # Estruturas leves para acumular dados
    todas_linhas = []
    todos_textos = []
    todos_retangulos = []
    todas_emendas = []
    pinos_mpu = []
    dados_por_pagina = {}   # apenas metadados, não imagens

    for i, page in enumerate(doc):
        if limite_paginas > 0 and i >= limite_paginas:
            logger.info(f"Limite de {limite_paginas} páginas atingido. Parando.", extra={'modulo': 'Roteador'})
            break

        try:
            logger.info(f">>> Extraindo página {i+1}/{total_paginas}", extra={'modulo': 'Roteador'})

            # Converter página para imagem
            pix = page.get_pixmap(dpi=300)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if DEBUG_MODE:
                cv2.imwrite(str(DEBUG_DIR / f"pagina_{i+1}_original.png"), img)

            # Restauração e detecção de símbolos
            esqueleto, binaria = restaurar_imagem(img)
            dados_pagina = processar_imagem_restaurada(esqueleto, binaria, original=img)

            # Extrair primitivas e acumular em listas planas
            linhas = dados_pagina.get('linhas', [])
            textos = dados_pagina.get('textos', [])
            retangulos = dados_pagina.get('retangulos', [])
            emendas = dados_pagina.get('curvas', [])

            todas_linhas.extend(linhas)
            todos_textos.extend(textos)
            todos_retangulos.extend(retangulos)
            todas_emendas.extend(emendas)

            # Guardar metadados por página (para futura referência)
            dados_por_pagina[i+1] = {
                'num_linhas': len(linhas),
                'num_textos': len(textos),
                'num_retangulos': len(retangulos),
                'num_emendas': len(emendas)
            }

            # Verificar Modo MPU
            resultado_mpu = processar_modo_mpu(dados_pagina, ferramenta='CarProg_A10')
            if resultado_mpu.get('modo') == 'mpu':
                logger.info(">>> MODO MPU ATIVADO – LEITURA DE MICROCONTROLADOR <<<", extra={'modulo': 'MPU'})
                pinos_mpu.extend(resultado_mpu.get('pinos', []))

            # Liberar memória local
            del img, esqueleto, binaria, dados_pagina

        except Exception as e:
            coletor.adicionar_erro(
                ErroPipeline(f"Erro na página {i+1}: {str(e)}", modulo='Roteador', causa=e),
                severidade=Severidade.MEDIA
            )
            continue

    doc.close()

    # Se alguma página ativou o MPU, retornar resultado especial
    if pinos_mpu:
        return {
            'status': 'ok',
            'modulo': 'MPU',
            'modo': 'mpu',
            'pinos': to_native(pinos_mpu),
            'num_pinos': len(pinos_mpu),
            'num_paginas': len(dados_por_pagina),
            'ferramenta': 'CarProg_A10'
        }

    # Verificar se há elementos gráficos suficientes para construir grafo
    total_retangulos = len(todos_retangulos)
    total_linhas = len(todas_linhas)

    if total_retangulos < 2 and total_linhas < 10:
        logger.warning("Poucos elementos gráficos - ativando modo DATASHEET/MANUAL", extra={'modulo': 'Roteador'})
        # Tentar extrair datasheet do próprio PDF
        try:
            pin_func = extrair_datasheet(caminho_pdf)
            if pin_func:
                logger.info(f"Datasheet extraído automaticamente: {len(pin_func)} funções", extra={'modulo': 'Roteador'})
                return {
                    'status': 'ok',
                    'modulo': 'M6',
                    'funcoes': to_native(pin_func),
                    'num_pinos': len(pin_func),
                    'num_paginas': len(dados_por_pagina),
                    'mensagem': 'Arquivo processado como datasheet (modo leve)'
                }
            else:
                return {
                    'status': 'erro',
                    'mensagem': 'Não foi possível extrair informações do arquivo. Verifique se é um diagrama ou datasheet válido.',
                    'num_paginas': len(dados_por_pagina)
                }
        except Exception as e:
            logger.error(f"Falha na extração automática de datasheet: {e}", extra={'modulo': 'Roteador'})
            return {
                'status': 'erro',
                'mensagem': f'Falha ao extrair datasheet: {str(e)}',
                'num_paginas': len(dados_por_pagina)
            }

    # Se chegou aqui, temos elementos gráficos suficientes -> construir grafo
    # Precisamos reconstruir a estrutura de dados por página para o grafo_rastreador
    # Mas não temos mais as imagens, apenas as primitivas acumuladas.
    # Para simplificar, vamos criar um dicionário com uma única página artificial contendo todas as primitivas.
    # NOTA: Isso perde a informação de página, mas é um trade-off para evitar memória.
    dados_consolidados = {
        1: {
            'linhas': todas_linhas,
            'textos': todos_textos,
            'retangulos': todos_retangulos,
            'curvas': todas_emendas,
            'canvas': (0, 0, 1000, 1000)  # dummy
        }
    }

    try:
        conexoes, G, pinos, perifs = processar_diagrama_multipagina(dados_consolidados)
        return {
            'status': 'ok',
            'modulo': 'M5',
            'conexoes': to_native(conexoes),
            'num_conexoes': len(conexoes),
            'num_pinos': len(pinos),
            'num_paginas': len(dados_por_pagina)
        }
    except Exception as e:
        logger.error(f"Erro ao construir grafo: {e}", extra={'modulo': 'Roteador'})
        return {
            'status': 'erro',
            'mensagem': f'Falha ao construir grafo: {str(e)}',
            'num_paginas': len(dados_por_pagina)
        }

# ============================================================
# DEMAIS FUNÇÕES (sem mudanças significativas)
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
    logger.info(f"Processando foto de celular: {caminho_imagem}", extra={'modulo': 'Roteador'})
    return processar_imagem_limpa(caminho_imagem)

def processar_manual(caminho_pdf):
    """Processamento específico para manuais/datasheets."""
    logger.info(f"Processando manual: {caminho_pdf}", extra={'modulo': 'Roteador'})
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

# ============================================================
# FUNÇÃO PRINCIPAL DO ROTEADOR
# ============================================================
@monitorar(modulo='Roteador')
def rotear_arquivo(caminho_arquivo, limite_paginas=0):
    """
    Classifica o arquivo e o encaminha para o módulo apropriado.
    Parâmetro limite_paginas: quantas páginas processar (0 = todas).
    """
    if not os.path.exists(caminho_arquivo):
        raise ErroPipeline(
            f"Arquivo não encontrado: {caminho_arquivo}",
            modulo='Roteador',
            severidade=Severidade.CRITICA
        )

    tipo = classificar_arquivo(caminho_arquivo)
    descricao = TIPOS_VALIDOS.get(tipo, 'Desconhecido')

    logger.info(f"Arquivo classificado como: {tipo} ({descricao})", extra={'modulo': 'Roteador'})

    funcao = ROTAS.get(tipo, processar_desconhecido)
    # Se a função aceitar limite_paginas, passar
    if tipo == 'pdf_rasterizado' or tipo == 'manual':
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
            elif resultado['resultado'].get('modulo') == 'M6':
                print(f"Funções de pinos extraídas: {resultado['resultado']['num_pinos']}")
            else:
                print(f"Conexões encontradas: {resultado['resultado'].get('num_conexoes', 0)}")
        else:
            print(f"Erro: {resultado['resultado'].get('mensagem', 'Desconhecido')}")
    except ErroPipeline as e:
        logger.critical(f"Falha crítica: {e.to_dict()}")
        sys.exit(1)
