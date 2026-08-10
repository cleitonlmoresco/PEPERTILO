"""
Módulo 8 - Interface Web (Servidor Flask) – versão com Blueprint.
Fornece upload, processamento assíncrono, visualização interativa,
correção manual, download de planilhas e busca inteligente.
"""

import os
import sys
import uuid
import json
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
JOB_EXPIRATION_DAYS = 7

def allowed_file(filename, allowed_extensions=None):
    if allowed_extensions is None:
        allowed_extensions = {'pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'webp'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def criar_job(arquivos, caminhos, nome_modulo='', caminho_datasheet=None, limite_paginas=0):
    job_id = str(uuid.uuid4())[:8]
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
    if job_id in jobs:
        jobs[job_id].update(kwargs)
        jobs[job_id]['ultima_atualizacao'] = datetime.now().isoformat()

def limpar_jobs_antigos():
    agora = datetime.now()
    expiracao = timedelta(days=JOB_EXPIRATION_DAYS)
    remover = []
    for jid, job in jobs.items():
        if job['status'] in ('concluido', 'erro'):
            criado = datetime.fromisoformat(job['criado_em'])
            if agora - criado > expiracao:
                remover.append(jid)
    for jid in remover:
        del jobs[jid]
        logger.info(f"Job {jid} removido por expiração")

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

            atualizar_job(job_id, progresso=50, etapa='Montando grafo')

            funcoes = None
            caminho_ds = job.get('caminho_datasheet')
            if caminho_ds and Path(caminho_ds).exists():
                from extracao_datasheet import extrair_datasheet
                try:
                    funcoes = extrair_datasheet(caminho_ds)
                except Exception as e:
                    logger.warning(f"Erro no datasheet: {e}")

            atualizar_job(job_id, progresso=80, etapa='Gerando planilha')

            conexoes = resultado_roteador['resultado'].get('conexoes', [])
            pinos_mpu = resultado_roteador['resultado'].get('pinos', [])
            modo = resultado_roteador['resultado'].get('modo', '')
            modulo = resultado_roteador['resultado'].get('modulo', '')

            resultado = {
                'conexoes': conexoes,
                'pinos': pinos_mpu,
                'funcoes': funcoes,
                'num_conexoes': len(conexoes),
                'num_pinos': len(pinos_mpu),
                'tipo': resultado_roteador['tipo'],
                'modulo_processado': modulo,
                'modo': modo,
                'num_paginas': resultado_roteador['resultado'].get('num_paginas', 0),
                'mensagem': resultado_roteador['resultado'].get('mensagem', '')
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

@bp.route('/upload', methods=['POST'])
def upload():
    if 'arquivos' not in request.files:
        return jsonify({'erro': 'Nenhum arquivo enviado'}), 400

    arquivos_enviados = request.files.getlist('arquivos')
    datasheet = request.files.get('datasheet')
    nome_modulo = request.form.get('nome_modulo', '').strip()
    limite_paginas = request.form.get('limite_paginas', 0, type=int)

    if not arquivos_enviados or all(f.filename == '' for f in arquivos_enviados):
        return jsonify({'erro': 'Nenhum arquivo selecionado'}), 400

    caminhos = []
    nomes_originais = []
    for file in arquivos_enviados:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            nome_unico = f"{uuid.uuid4().hex[:8]}_{filename}"
            caminho = UPLOAD_FOLDER / nome_unico
            file.save(str(caminho))
            caminhos.append(str(caminho))
            nomes_originais.append(file.filename)
            logger.info(f"Arquivo salvo: {caminho}")

    caminho_ds = None
    if datasheet and datasheet.filename and allowed_file(datasheet.filename):
        filename = secure_filename(datasheet.filename)
        nome_ds = f"ds_{uuid.uuid4().hex[:8]}_{filename}"
        caminho_ds = UPLOAD_FOLDER / nome_ds
        datasheet.save(str(caminho_ds))
        caminho_ds = str(caminho_ds)
        logger.info(f"Datasheet salvo: {caminho_ds}")

    if not caminhos:
        return jsonify({'erro': 'Nenhum arquivo válido'}), 400

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

@bp.route('/api/conexoes/<job_id>')
def api_conexoes(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'erro': 'Job não encontrado'}), 404

    resultado = job.get('resultado', {})
    funcoes = resultado.get('funcoes') or {}

    conexoes = []
    for conn in resultado.get('conexoes', []):
        conexoes.append({
            'pino': conn[0] if len(conn) > 0 else '',
            'destino': conn[1] if len(conn) > 1 else '',
            'cor': conn[2] if len(conn) > 2 else '',
            'bitola': conn[3] if len(conn) > 3 else '',
            'funcao': funcoes.get(conn[0], 'Desconhecida'),
            'confianca': 85
        })
    return jsonify({'conexoes': conexoes, 'total': len(conexoes)})

@bp.route('/api/corrigir/<job_id>', methods=['POST'])
def api_corrigir(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'erro': 'Job não encontrado'}), 404

    dados = request.get_json()
    indice = dados.get('indice')
    campo = dados.get('campo')
    valor = dados.get('valor')

    try:
        resultado = job.get('resultado', {})
        conexoes = resultado.get('conexoes', [])
        if 0 <= indice < len(conexoes):
            conn = list(conexoes[indice])
            mapeamento = {'pino': 0, 'destino': 1, 'cor': 2, 'bitola': 3}
            if campo in mapeamento:
                conn[mapeamento[campo]] = valor
                conexoes[indice] = tuple(conn)
                resultado['conexoes'] = conexoes
                resultado['corrigido_manualmente'] = True
                job['resultado'] = resultado
                logger.info(f"Correção aplicada: job={job_id}, índice={indice}, campo={campo}")
                return jsonify({'status': 'ok'})
        return jsonify({'erro': 'Índice ou campo inválido'}), 400
    except Exception as e:
        return jsonify({'erro': str(e)}), 500

@bp.route('/download/<job_id>')
def download_planilha(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'erro': 'Job não encontrado'}), 404

    resultado = job.get('resultado', {})

    if resultado.get('modo') == 'mpu':
        import csv
        import io
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
        nome_modulo = job.get('nome_modulo', 'MPU').replace(' ', '_')
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'pinos_{nome_modulo}.csv'
        )

    conexoes = resultado.get('conexoes', [])
    if not conexoes:
        return jsonify({'erro': 'Nenhuma conexão para exportar'}), 400

    from consolidacao_exportacao import consolidar_conexoes, gerar_excel
    pin_func = resultado.get('funcoes', None)
    df = consolidar_conexoes(conexoes, pin_func)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
    gerar_excel(df, tmp.name)

    @current_app.after_request
    def cleanup(response):
        try:
            os.remove(tmp.name)
        except Exception as e:
            logger.warning(f"Erro ao remover arquivo temporário: {e}")
        return response

    nome_modulo = job.get('nome_modulo', 'ECU').replace(' ', '_')
    return send_file(tmp.name,
                     as_attachment=True,
                     download_name=f'pinagem_{nome_modulo}.xlsx')

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

if __name__ == '__main__':
    from flask import Flask
    app = Flask(__name__)
    app.register_blueprint(bp, url_prefix='/')
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'chave-secreta-oficina-2025')
    logger.info("Servidor web iniciado (standalone)", extra={'modulo': 'M8', 'funcao': 'main'})
    app.run(debug=True, host='0.0.0.0', port=5000)
