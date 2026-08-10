"""
Módulo 1 - Classificação de Arquivos.
Identifica o tipo de arquivo (PDF vetorial, PDF rasterizado, imagem, foto, manual).
"""

import os
import fitz
from PIL import Image
from logger_erros import logger, monitorar, ErroPipeline, Severidade, tratar_erro_controlado

TIPOS_VALIDOS = {
    'pdf_vetorial': 'Diagrama em PDF com camada vetorial',
    'pdf_rasterizado': 'PDF escaneado (imagem)',
    'imagem_limpa': 'Imagem digitalizada ou captura limpa',
    'foto_celular': 'Foto de celular com distorções',
    'manual': 'Documento de texto com tabelas (manual/datasheet)',
    'desconhecido': 'Tipo não identificado'
}

def get_tipo_arquivo(caminho):
    ext = os.path.splitext(caminho)[1].lower()
    if ext == '.pdf':
        return 'pdf'
    elif ext in ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp'):
        return 'imagem'
    else:
        logger.warning(f"Extensão não reconhecida: {ext}", extra={'modulo': 'M1', 'funcao': 'get_tipo_arquivo'})
        return 'desconhecido'

@monitorar(modulo='M1')
def possui_camada_vetorial(caminho_pdf, limiar_linhas=50):
    if not os.path.exists(caminho_pdf):
        raise ErroPipeline(f"Arquivo não encontrado: {caminho_pdf}", modulo='M1', severidade=Severidade.ALTA)

    try:
        doc = fitz.open(caminho_pdf)
    except Exception as e:
        raise ErroPipeline(f"Não foi possível abrir o PDF: {str(e)}", modulo='M1', severidade=Severidade.ALTA, causa=e)

    total_linhas = 0
    paginas_com_dados = 0
    try:
        for pagina in doc:
            paths = pagina.get_drawings()
            num_linhas = sum(1 for path in paths if path.get('type') == 'l')
            total_linhas += num_linhas
            if num_linhas > 0:
                paginas_com_dados += 1
    except Exception as e:
        logger.error(f"Erro ao analisar PDF: {str(e)}", extra={'modulo': 'M1'})
        doc.close()
        return False
    finally:
        doc.close()

    if paginas_com_dados == 0:
        logger.info("PDF não contém páginas com dados vetoriais", extra={'modulo': 'M1'})
        return False

    media = total_linhas / paginas_com_dados
    resultado = media >= limiar_linhas
    logger.debug(f"Média de linhas por página: {media:.1f} (limiar={limiar_linhas}) -> vetorial={resultado}", extra={'modulo': 'M1'})
    return resultado

@monitorar(modulo='M1')
def analisar_imagem(caminho_imagem):
    if not os.path.exists(caminho_imagem):
        raise ErroPipeline(f"Imagem não encontrada: {caminho_imagem}", modulo='M1', severidade=Severidade.ALTA)

    with Image.open(caminho_imagem) as img:
        largura, altura = img.size
        proporcao = largura / altura
        exif = img._getexif() if hasattr(img, '_getexif') else None
        tem_exif = exif is not None
        fabricante = ''
        if tem_exif:
            fabricante = exif.get(271, '')

        provavel_foto = False
        if tem_exif and any(marca in fabricante.lower() for marca in ['samsung', 'apple', 'xiaomi', 'motorola', 'lg']):
            provavel_foto = True
            logger.debug(f"EXIF indica foto: fabricante={fabricante}", extra={'modulo': 'M1'})
        if abs(proporcao - 1.33) < 0.1 or abs(proporcao - 1.78) < 0.1:
            provavel_foto = True
            logger.debug(f"Proporção indica foto: {proporcao:.2f}", extra={'modulo': 'M1'})

    return {
        'largura': largura, 'altura': altura, 'proporcao': proporcao,
        'tem_exif': tem_exif, 'fabricante': fabricante, 'provavel_foto': provavel_foto
    }

def analisar_conteudo_pdf(caminho_pdf):
    """Pré-classificação: distingue diagrama de manual/documento texto."""
    try:
        doc = fitz.open(caminho_pdf)
        total_texto = 0
        total_drawings = 0
        for page in doc:
            total_texto += len(page.get_text().strip())
            total_drawings += len(page.get_drawings())
        doc.close()
        if total_texto == 0 and total_drawings == 0:
            return 'vazio'
        if total_drawings > 0 and total_texto / (total_drawings + 1) < 50:  # heurística
            return 'diagrama'
        else:
            return 'manual'
    except Exception as e:
        logger.warning(f"Erro na pré-classificação: {e}", extra={'modulo': 'M1'})
        return 'desconhecido'

@monitorar(modulo='M1')
def classificar_arquivo(caminho_arquivo):
    if not os.path.exists(caminho_arquivo):
        raise ErroPipeline(f"Arquivo não encontrado: {caminho_arquivo}", modulo='M1', severidade=Severidade.CRITICA)

    tipo_base = get_tipo_arquivo(caminho_arquivo)
    logger.info(f"Classificando arquivo: {os.path.basename(caminho_arquivo)} (tipo base: {tipo_base})", extra={'modulo': 'M1'})

    if tipo_base == 'pdf':
        if possui_camada_vetorial(caminho_arquivo):
            return 'pdf_vetorial'
        else:
            # Pré-classificação para decidir entre rasterizado (diagrama) ou manual
            categoria = analisar_conteudo_pdf(caminho_arquivo)
            if categoria == 'manual':
                logger.info("PDF classificado como manual/datasheet", extra={'modulo': 'M1'})
                return 'manual'
            else:
                return 'pdf_rasterizado'
    elif tipo_base == 'imagem':
        info, erro = tratar_erro_controlado(analisar_imagem, caminho_arquivo, valor_padrao={'provavel_foto': False}, modulo='M1')
        if erro:
            logger.warning(f"Erro ao analisar imagem, assumindo digitalização limpa: {erro}", extra={'modulo': 'M1'})
        if info.get('provavel_foto'):
            return 'foto_celular'
        else:
            return 'imagem_limpa'
    else:
        logger.warning(f"Tipo de arquivo não reconhecido: {caminho_arquivo}", extra={'modulo': 'M1'})
        return 'desconhecido'

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Uso: python classificador.py <arquivo>")
        sys.exit(1)
    try:
        resultado = classificar_arquivo(sys.argv[1])
        print(f"Classificação: {resultado} ({TIPOS_VALIDOS[resultado]})")
    except ErroPipeline as e:
        logger.critical(f"Falha na classificação: {e.to_dict()}")
        sys.exit(1)
