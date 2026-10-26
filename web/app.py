"""
Módulo 8 - Interface Web (Servidor Flask) – versão com Blueprint.
Fornece upload, processamento assíncrono, visualização interativa,
correção manual, download de planilhas e busca inteligente.
"""

import os
import sys
import io
import csv
import uuid
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path

from flask import (Blueprint, render_template, request, jsonify, send_file,
                   redirect, url_for, current_app)
from werkzeug.utils import secure_filename

sys.path.insert(0, str(Path(__file__).parent.parent))

from logger_erros import logger, ErroPipeline, Severidade
from roteador import rotear_arquivo, TIPOS_VALIDOS

bp = Blueprint('modulo8', __name__, template_folder='templates')

def get_config(key, default):
    return current_app.config.get(key, default) if current_app else default

UPLOAD_FOLDER = Path(__file__).parent / 'uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)

jobs = {}
jobs_lock = threading.Lock()
JOB_EXPIRATION_DAYS = 7

# Limites de upload (evitam encher o disco e travam o processo com arquivos gigantes)
MAX_FILE_MB = int(os.environ.get('MAX_FILE_MB', '25'))
MAX_TOTAL_MB = int(os.environ.get('MAX_TOTAL_MB', '100'))
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024
MAX_TOTAL_BYTES = MAX_TOTAL_MB * 1024 * 1024

def allowed_file(filename, allowed_extensions=None):
    if allowed_extensions is None:
        allowed_extensions = {'pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'webp'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def criar_job(arquivos, caminhos, nome_modulo='', caminho_datasheet=None, limite_paginas=0):
    job_id = str(uuid.uuid4())[:8]
    with jobs_lock:
        jobs[job_id] = {
            'id': job_id,
            'arquivos': arquivos,
            'caminhos': caminhos,
            'caminho_datasheet': caminho_datasheet,
            'nome_modulo': nome_modulo,
            'limite_paginas': limite_paginas,
            'status': 'aguardando',
            'progresso': 0,
            'etapa': 'Na fila',
            'resultado': None,
            'erros': [],
            'criado_em': datetime.now().isoformat(),
            'concluido_em': None,
            'ultima_atualizacao': datetime.now().isoformat()
        }
    return job_id

def atualizar_job(job_id, **kwargs):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(kwargs)
            jobs[job_id]['ultima_atualizacao'] = datetime.now().isoformat()

def _remover_arquivos_do_job(job):
    """Apaga os arquivos enviados de um job (evita crescimento infinito de uploads/)."""
    caminhos = list(job.get('caminhos') or [])
    if job.get('caminho_datasheet'):
        caminhos.append(job['caminho_datasheet'])
    for caminho in caminhos:
        try:
            Path(caminho).unlink(missing_ok=True)
        except OSError as e:
            logger.warning(f"Não foi possível remover {caminho}: {e}")

def limpar_jobs_antigos():
    agora = datetime.now()
    expiracao = timedelta(days=JOB_EXPIRATION_DAYS)
    with jobs_lock:
        remover = [
            jid for jid, job in jobs.items()
            if job['status'] in ('concluido', 'erro')
            and agora - datetime.fromisoformat(job['criado_em']) > expiracao
        ]
        jobs_removidos = [jobs.pop(jid) for jid in remover]
    for job in jobs_removidos:
        _remover_arquivos_do_job(job)
        logger.info(f"Job {job['id']} removido por expiração")

def processar_job_assincrono(job_id):
    def tarefa():
        job = jobs.get(job_id)
        if not job:
            return
        try:
            atualizar_job(job_id, status='processando', etapa='Classificando arquivo')

            caminhos = job.get('caminhos', [])
            if not caminhos:
                raise ErroPipeline("Nenhum arquivo para processar", severidade=Severidade.CRITICA)

            caminho_principal = caminhos[0]
            if not Path(caminho_principal).exists():
                raise ErroPipeline(f"Arquivo não encontrado: {caminho_principal}", severidade=Severidade.CRITICA)

            limite = job.get('limite_paginas', 0)

            atualizar_job(job_id, progresso=10, etapa='Extraindo dados')
            resultado_roteador = rotear_arquivo(caminho_principal, limite_paginas=limite)

            if resultado_roteador['resultado']['status'] != 'ok':
                raise ErroPipeline(
                    resultado_roteador['resultado'].get('mensagem', 'Erro desconhecido'),
                    modulo=resultado_roteador['tipo'],
                    severidade=Severidade.ALTA
                )

            # Captura TODOS os campos do resultado do roteador
            resultado_roteador_result = resultado_roteador['resultado']
            conexoes = resultado_roteador_result.get('conexoes') or []
            pinos_mpu = resultado_roteador_result.get('pinos') or []
            funcoes_da_extracao = resultado_roteador_result.get('funcoes') or {}

            # Se houver um datasheet enviado separadamente, pode complementar ou substituir
            funcoes = dict(funcoes_da_extracao)
            caminho_ds = job.get('caminho_datasheet')
            if caminho_ds and Path(caminho_ds).exists():
                from extracao_datasheet import extrair_datasheet
                try:
                    funcoes_ds = extrair_datasheet(caminho_ds)
                    if funcoes_ds:
                        funcoes.update(funcoes_ds)  # mescla com as extraídas do PDF
                except Exception as e:
                    logger.warning(f"Erro no datasheet: {e}")

            atualizar_job(job_id, progresso=80, etapa='Gerando planilha')

            modo = resultado_roteador_result.get('modo', '')
            modulo = resultado_roteador_result.get('modulo', '')

            resultado = {
                'conexoes': conexoes,
                'pinos': pinos_mpu,
                'funcoes': funcoes,
                'num_conexoes': len(conexoes),
                'num_pinos': len(pinos_mpu) or len(funcoes) or resultado_roteador_result.get('num_pinos', 0),
                'tipo': resultado_roteador['tipo'],
                'modulo_processado': modulo,
                'modo': modo,
                'num_paginas': resultado_roteador_result.get('num_paginas', 0),
                'mensagem': resultado_roteador_result.get('mensagem', '')
            }

            atualizar_job(job_id,
                          status='concluido',
                          progresso=100,
                          etapa='Finalizado',
                          resultado=resultado,
                          concluido_em=datetime.now().isoformat())

        except ErroPipeline as e:
            logger.error(f"Job {job_id} falhou: {e.mensagem}")
            atualizar_job(job_id,
                          status='erro',
                          etapa=f'Falha: {e.mensagem[:50]}',
                          erros=[e.to_dict()])
        except Exception as e:
            logger.critical(f"Job {job_id} erro inesperado: {str(e)}", exc_info=True)
            atualizar_job(job_id,
                          status='erro',
                          etapa='Erro inesperado',
                          erros=[{'mensagem': str(e), 'severidade': 'critica'}])

    thread = threading.Thread(target=tarefa, daemon=True)
    thread.start()

# Rotas
@bp.route('/')
def index():
    limpar_jobs_antigos()
    return render_template('index.html',
                           jobs=list(jobs.values())[-20:],
                           tipos=TIPOS_VALIDOS)

@bp.route('/carregamento/<job_id>')
def carregamento(job_id):
    job = jobs.get(job_id)
    if not job:
        return redirect(url_for('modulo8.index'))
    return render_template('carregamento.html', job_id=job_id, job=job)

def _tamanho_stream(file):
    """Retorna o tamanho do upload sem consumir o stream."""
    pos = file.stream.tell()
    file.stream.seek(0, os.SEEK_END)
    tamanho = file.stream.tell()
    file.stream.seek(pos)
    return tamanho

def _salvar_upload(file, prefixo=''):
    """Valida e grava um upload. Retorna (Path|None, erro|None)."""
    if not file or not file.filename:
        return None, 'arquivo sem nome'
    if not allowed_file(file.filename):
        return None, f'extensão não permitida ({Path(file.filename).suffix})'
    if _tamanho_stream(file) > MAX_FILE_BYTES:
        return None, f'arquivo maior que {MAX_FILE_MB} MB'
    filename = secure_filename(file.filename) or f"arquivo{Path(file.filename).suffix.lower()}"
    destino = UPLOAD_FOLDER / f"{prefixo}{uuid.uuid4().hex[:8]}_{filename}"
    file.save(str(destino))
    return destino, None

@bp.route('/upload', methods=['POST'])
def upload():
    if 'arquivos' not in request.files:
        return jsonify({'erro': 'Nenhum arquivo enviado'}), 400

    if request.content_length and request.content_length > MAX_TOTAL_BYTES:
        return jsonify({'erro': f'Envio muito grande (máximo {MAX_TOTAL_MB} MB).'}), 413

    arquivos_enviados = request.files.getlist('arquivos')
    datasheet = request.files.get('datasheet')
    nome_modulo = request.form.get('nome_modulo', '').strip()[:120]
    limite_paginas = request.form.get('limite_paginas', 0, type=int)

    if not arquivos_enviados or all(f.filename == '' for f in arquivos_enviados):
        return jsonify({'erro': 'Nenhum arquivo selecionado'}), 400

    caminhos = []
    nomes_originais = []
    erros = []
    for file in arquivos_enviados:
        if not file or not file.filename:
            continue
        caminho, erro = _salvar_upload(file)
        if erro:
            erros.append(f"{file.filename}: {erro}")
            continue
        caminhos.append(str(caminho))
        nomes_originais.append(file.filename)
        logger.info(f"Arquivo salvo: {caminho}")

    caminho_ds = None
    if datasheet and datasheet.filename:
        destino_ds, erro_ds = _salvar_upload(datasheet, prefixo='ds_')
        if erro_ds:
            erros.append(f"datasheet: {erro_ds}")
        else:
            caminho_ds = str(destino_ds)
            logger.info(f"Datasheet salvo: {caminho_ds}")

    if not caminhos:
        if caminho_ds:
            Path(caminho_ds).unlink(missing_ok=True)
        detalhe = f" Detalhes: {'; '.join(erros)}" if erros else ''
        return jsonify({'erro': f'Nenhum arquivo válido.{detalhe}'}), 400

    job_id = criar_job(
        arquivos=nomes_originais,
        caminhos=caminhos,
        nome_modulo=nome_modulo,
        caminho_datasheet=caminho_ds,
        limite_paginas=limite_paginas
    )

    processar_job_assincrono(job_id)
    return jsonify({'job_id': job_id, 'status': 'iniciado'})

@bp.route('/status/<job_id>')
def status_job(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'erro': 'Job não encontrado'}), 404
    return jsonify({
        'id': job['id'],
        'status': job['status'],
        'progresso': job['progresso'],
        'etapa': job['etapa'],
        'erros': job.get('erros', [])
    })

@bp.route('/jobs')
def listar_jobs():
    limpar_jobs_antigos()
    return jsonify({jid: {
        'status': j['status'],
        'progresso': j['progresso'],
        'criado_em': j['criado_em'],
        'concluido_em': j['concluido_em']
    } for jid, j in jobs.items()})

@bp.route('/resultado/<job_id>')
def ver_resultado(job_id):
    job = jobs.get(job_id)
    if not job or job['status'] != 'concluido':
        return redirect(url_for('modulo8.index'))

    resultado = job.get('resultado', {})
    if resultado.get('modo') == 'mpu':
        return render_template('resultado_mpu.html', job=job, resultado=resultado)
    return render_template('resultado.html', job=job, resultado=resultado)

CAMPOS_CONEXAO = ('pino', 'destino', 'cor', 'bitola')

def _conexao_para_dict(conn):
    """Normaliza uma conexão (tupla, lista ou dict) para um dict com as 4 chaves."""
    if isinstance(conn, dict):
        return {campo: conn.get(campo, '') for campo in CAMPOS_CONEXAO}
    if isinstance(conn, (list, tuple)):
        return {campo: (conn[i] if i < len(conn) else '') for i, campo in enumerate(CAMPOS_CONEXAO)}
    return {campo: '' for campo in CAMPOS_CONEXAO}

@bp.route('/api/conexoes/<job_id>')
def api_conexoes(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'erro': 'Job não encontrado'}), 404

    resultado = job.get('resultado') or {}
    funcoes = resultado.get('funcoes') or {}

    conexoes = []
    for conn in resultado.get('conexoes') or []:
        dados = _conexao_para_dict(conn)
        dados['funcao'] = funcoes.get(dados['pino'], 'Desconhecida')
        dados['confianca'] = 85
        conexoes.append(dados)
    return jsonify({'conexoes': conexoes, 'total': len(conexoes)})

@bp.route('/api/corrigir/<job_id>', methods=['POST'])
def api_corrigir(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'erro': 'Job não encontrado'}), 404
    if job.get('status') != 'concluido':
        return jsonify({'erro': 'Job ainda não foi concluído'}), 409

    dados = request.get_json(silent=True)
    if not isinstance(dados, dict):
        return jsonify({'erro': 'Corpo JSON inválido'}), 400

    mapeamento = {campo: i for i, campo in enumerate(CAMPOS_CONEXAO)}
    campo = dados.get('campo')
    valor = dados.get('valor')
    try:
        indice = int(dados.get('indice'))
    except (TypeError, ValueError):
        return jsonify({'erro': 'Índice inválido'}), 400

    if campo not in mapeamento:
        return jsonify({'erro': f'Campo inválido (use: {", ".join(mapeamento)})'}), 400
    if not isinstance(valor, str) or len(valor) > 200:
        return jsonify({'erro': 'Valor inválido'}), 400

    with jobs_lock:
        resultado = job.get('resultado') or {}
        conexoes = list(resultado.get('conexoes') or [])
        if not 0 <= indice < len(conexoes):
            return jsonify({'erro': 'Índice fora do intervalo'}), 400

        conexao = _conexao_para_dict(conexoes[indice])
        conexao[campo] = valor
        conexoes[indice] = tuple(conexao[c] for c in CAMPOS_CONEXAO)
        resultado['conexoes'] = conexoes
        resultado['corrigido_manualmente'] = True
        job['resultado'] = resultado

    logger.info(f"Correção aplicada: job={job_id}, índice={indice}, campo={campo}")
    return jsonify({'status': 'ok'})

@bp.route('/download/<job_id>')
def download_planilha(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'erro': 'Job não encontrado'}), 404

    resultado = job.get('resultado', {})

    if resultado.get('modo') == 'mpu':
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Pino', 'Lado', 'Função', 'Cor do Fio', 'Ferramenta', 'Confiança'])
        for pino in resultado.get('pinos', []):
            writer.writerow([
                pino.get('pino', ''),
                pino.get('lado', ''),
                pino.get('funcao', ''),
                pino.get('cor_fio', ''),
                pino.get('ferramenta', ''),
                f"{pino.get('confianca', 0)}%"
            ])
        output.seek(0)
        nome_modulo = secure_filename(job.get('nome_modulo') or '') or 'MPU'
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8-sig')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'pinos_{nome_modulo}.csv'
        )

    # Para diagramas: gera Excel com conexões e funções
    conexoes = resultado.get('conexoes', [])
    if not conexoes:
        # Se não houver conexões, mas houver funções, gerar CSV com funções
        funcoes = resultado.get('funcoes', {})
        if funcoes:
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(['Pino', 'Função'])
            for pino, func in funcoes.items():
                writer.writerow([pino, func])
            output.seek(0)
            nome_modulo = secure_filename(job.get('nome_modulo') or '') or 'Manual'
            return send_file(
                io.BytesIO(output.getvalue().encode('utf-8-sig')),
                mimetype='text/csv',
                as_attachment=True,
                download_name=f'funcoes_{nome_modulo}.csv'
            )
        return jsonify({'erro': 'Nenhum dado para exportar'}), 400

    from consolidacao_exportacao import consolidar_conexoes, gerar_excel
    pin_func = resultado.get('funcoes', None)
    df = consolidar_conexoes(conexoes, pin_func)

    # Fecha o handle antes de escrever: no Windows, um NamedTemporaryFile aberto
    # bloqueia a escrita no mesmo caminho (era a causa de PermissionError no download).
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
    caminho_tmp = tmp.name
    tmp.close()
    try:
        gerar_excel(df, caminho_tmp)
    except Exception:
        Path(caminho_tmp).unlink(missing_ok=True)
        raise

    # Lê para memória e apaga o temporário na hora: no Windows o send_file
    # mantém o handle aberto e o after_request falhava com WinError 32
    # (arquivo vazava em %TEMP% a cada download).
    try:
        conteudo = Path(caminho_tmp).read_bytes()
    finally:
        Path(caminho_tmp).unlink(missing_ok=True)

    nome_modulo = secure_filename(job.get('nome_modulo') or '') or 'ECU'
    return send_file(
        io.BytesIO(conteudo),
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'pinagem_{nome_modulo}.xlsx'
    )

@bp.route('/busca')
def busca():
    return render_template('busca.html')

@bp.route('/api/buscar')
def api_buscar():
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'resultados': [], 'total': 0})

    resultados = []
    for job in jobs.values():
        if job['status'] == 'concluido':
            nome = job.get('nome_modulo', '')
            if query.lower() in nome.lower():
                resultados.append({
                    'id': job['id'],
                    'nome': nome,
                    'data': job['concluido_em'],
                    'status': 'Concluído'
                })
    return jsonify({'resultados': resultados, 'total': len(resultados)})

def create_app():
    """Fábrica da aplicação (permite testar com o test client do Flask)."""
    from flask import Flask
    app = Flask(__name__)
    app.register_blueprint(bp, url_prefix='/')
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or os.urandom(24).hex()
    app.config['MAX_CONTENT_LENGTH'] = MAX_TOTAL_BYTES

    @app.errorhandler(413)
    def _arquivo_grande(_erro):
        return jsonify({'erro': f'Envio muito grande (máximo {MAX_TOTAL_MB} MB).'}), 413

    return app

if __name__ == '__main__':
    app = create_app()
    host = os.environ.get('HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', '5000'))
    debug = os.environ.get('FLASK_DEBUG', '').lower() in ('1', 'true', 'yes')
    logger.info("Servidor web iniciado (standalone)", extra={'modulo': 'M8', 'funcao': 'main'})
    if debug and host not in ('127.0.0.1', 'localhost'):
        logger.warning(
            "FLASK_DEBUG ativo com o servidor exposto na rede — o debugger do Werkzeug "
            "permite execução remota de código. Use apenas em localhost.",
            extra={'modulo': 'M8'}
        )
    app.run(debug=debug, host=host, port=port, threaded=True)
