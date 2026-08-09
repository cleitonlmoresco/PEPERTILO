"""
Módulo 8 - Interface Web (Servidor Flask).
Fornece upload, processamento assíncrono, visualização interativa,
correção manual, download de planilhas e busca inteligente.
"""

import os
import sys
import uuid
import json
import tempfile
import threading
from datetime import datetime
from pathlib import Path

from flask import (Flask, render_template, request, jsonify, send_file,
                   redirect, url_for)
from werkzeug.utils import secure_filename

# Adiciona o diretório raiz ao path para importar os módulos do pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))

from logger_erros import logger, ErroPipeline, Severidade
from roteador import rotear_arquivo, TIPOS_VALIDOS

# ------------------------------------------------------------
# Configuração da aplicação
# ------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'chave-secreta-oficina-2025')

UPLOAD_FOLDER = Path(__file__).parent / 'uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'webp'}

# Armazenamento de jobs em memória (em produção, usar banco de dados)
jobs = {}

# ------------------------------------------------------------
# Funções auxiliares
# ------------------------------------------------------------
def allowed_file(filename):
    """Verifica se a extensão do arquivo é permitida."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def criar_job(arquivos, nome_modulo=''):
    """Cria um registro de job para acompanhamento."""
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        'id': job_id,
        'arquivos': arquivos,
        'nome_modulo': nome_modulo,
        'status': 'aguardando',
        'progresso': 0,
        'etapa': 'Na fila',
        'resultado': None,
        'erros': [],
        'criado_em': datetime.now().isoformat(),
        'concluido_em': None
    }
    return job_id

def atualizar_job(job_id, **kwargs):
    """Atualiza os campos de um job."""
    if job_id in jobs:
        jobs[job_id].update(kwargs)

# ------------------------------------------------------------
# Processamento assíncrono do job
# ------------------------------------------------------------
def processar_job_assincrono(job_id):
    """
    Processa o job em background usando thread separada.
    """
    def tarefa():
        job = jobs.get(job_id)
        if not job:
            return
        try:
            atualizar_job(job_id, status='processando', etapa='Classificando arquivo')

            caminho_principal = job['caminhos'][0] if job.get('caminhos') else None
            if not caminho_principal:
                raise ErroPipeline("Nenhum arquivo para processar", severidade=Severidade.CRITICA)

            # Módulo 1 - Roteamento
            atualizar_job(job_id, progresso=10, etapa='Extraindo dados')
            resultado_roteador = rotear_arquivo(caminho_principal)

            if resultado_roteador['resultado']['status'] != 'ok':
                raise ErroPipeline(
                    resultado_roteador['resultado'].get('mensagem', 'Erro desconhecido'),
                    modulo=resultado_roteador['tipo'],
                    severidade=Severidade.ALTA
                )

            atualizar_job(job_id, progresso=50, etapa='Montando grafo')

            # Se houver datasheet, processar Módulo 6
            funcoes = None
            if job.get('caminho_datasheet'):
                from extracao_datasheet import extrair_datasheet
                try:
                    funcoes = extrair_datasheet(job['caminho_datasheet'])
                except Exception as e:
                    logger.warning(f"Erro no datasheet: {e}")

            atualizar_job(job_id, progresso=80, etapa='Gerando planilha')

            # Consolidar resultado
            conexoes = resultado_roteador['resultado'].get('conexoes', [])
            pinos_mpu = resultado_roteador['resultado'].get('pinos', [])
            modo = resultado_roteador['resultado'].get('modo', '')

            resultado = {
                'conexoes': conexoes,
                'pinos': pinos_mpu,
                'funcoes': funcoes,
                'num_conexoes': len(conexoes),
                'num_pinos': len(pinos_mpu),
                'tipo': resultado_roteador['tipo'],
                'modulo_processado': resultado_roteador['resultado'].get('modulo'),
                'modo': modo
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
            logger.critical(f"Job {job_id} erro inesperado: {str(e)}")
            atualizar_job(job_id,
                          status='erro',
                          etapa='Erro inesperado',
                          erros=[{'mensagem': str(e), 'severidade': 'critica'}])

    thread = threading.Thread(target=tarefa, daemon=True)
    thread.start()

# ------------------------------------------------------------
# Rotas da aplicação
# ------------------------------------------------------------
@app.route('/')
def index():
    """Dashboard principal."""
    return render_template('index.html',
                           jobs=list(jobs.values())[-20:],
                           tipos=TIPOS_VALIDOS)

@app.route('/upload', methods=['POST'])
def upload():
    """Endpoint de upload de arquivos."""
    if 'arquivos' not in request.files:
        return jsonify({'erro': 'Nenhum arquivo enviado'}), 400

    arquivos_enviados = request.files.getlist('arquivos')
    datasheet = request.files.get('datasheet')
    nome_modulo = request.form.get('nome_modulo', '').strip()

    if not arquivos_enviados or all(f.filename == '' for f in arquivos_enviados):
        return jsonify({'erro': 'Nenhum arquivo selecionado'}), 400

    # Validar e salvar arquivos
    caminhos = []
    for file in arquivos_enviados:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            nome_unico = f"{uuid.uuid4().hex[:8]}_{filename}"
            caminho = UPLOAD_FOLDER / nome_unico
            file.save(str(caminho))
            caminhos.append({
                'original': file.filename,
                'caminho': str(caminho),
                'tamanho': os.path.getsize(caminho)
            })

    caminho_ds = None
    if datasheet and datasheet.filename and allowed_file(datasheet.filename):
        filename = secure_filename(datasheet.filename)
        nome_ds = f"ds_{uuid.uuid4().hex[:8]}_{filename}"
        caminho_ds = UPLOAD_FOLDER / nome_ds
        datasheet.save(str(caminho_ds))
        caminho_ds = str(caminho_ds)

    if not caminhos:
        return jsonify({'erro': 'Nenhum arquivo válido'}), 400

    job_id = criar_job([c['original'] for c in caminhos], nome_modulo)
    jobs[job_id]['caminhos'] = [c['caminho'] for c in caminhos]
    jobs[job_id]['caminho_datasheet'] = caminho_ds

    processar_job_assincrono(job_id)

    return jsonify({'job_id': job_id, 'status': 'iniciado'})

@app.route('/status/<job_id>')
def status_job(job_id):
    """Retorna o status atual de um job (para polling via AJAX)."""
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

@app.route('/resultado/<job_id>')
def ver_resultado(job_id):
    """Página de visualização interativa do resultado."""
    job = jobs.get(job_id)
    if not job or job['status'] != 'concluido':
        return redirect(url_for('index'))

    resultado = job.get('resultado', {})

    # Verificar se é resultado MPU
    if resultado.get('modo') == 'mpu':
        return render_template('resultado_mpu.html', job=job, resultado=resultado)

    return render_template('resultado.html', job=job, resultado=resultado)

@app.route('/api/conexoes/<job_id>')
def api_conexoes(job_id):
    """Retorna as conexões em JSON para o overlay interativo."""
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

@app.route('/api/corrigir/<job_id>', methods=['POST'])
def api_corrigir(job_id):
    """Recebe correções manuais do visualizador e atualiza o resultado."""
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

@app.route('/download/<job_id>')
def download_planilha(job_id):
    """Gera e envia a planilha Excel do resultado."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({'erro': 'Job não encontrado'}), 404

    resultado = job.get('resultado', {})

    # Se for resultado MPU, gerar CSV dos pinos
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

    # Resultado normal de diagrama elétrico
    conexoes = resultado.get('conexoes', [])
    if not conexoes:
        return jsonify({'erro': 'Nenhuma conexão para exportar'}), 400

    from consolidacao_exportacao import consolidar_conexoes, gerar_excel

    pin_func = resultado.get('funcoes', None)
    df = consolidar_conexoes(conexoes, pin_func)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
    gerar_excel(df, tmp.name)

    nome_modulo = job.get('nome_modulo', 'ECU').replace(' ', '_')
    return send_file(tmp.name,
                     as_attachment=True,
                     download_name=f'pinagem_{nome_modulo}.xlsx')

@app.route('/busca')
def busca():
    """Página de busca inteligente (RAG)."""
    return render_template('busca.html')

@app.route('/api/buscar')
def api_buscar():
    """Endpoint de busca semântica (placeholder para RAG)."""
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'resultados': [], 'total': 0})

    resultados = []
    for job in jobs.values():
        if job['status'] == 'concluido':
            nome = job.get('nome_modulo', '')
            if query.lower() in nome.lower():
                resultados.append({
                    'job_id': job['id'],
                    'nome': nome,
                    'data': job['concluido_em'],
                    'num_conexoes': len(job.get('resultado', {}).get('conexoes', []))
                })
    return jsonify({'resultados': resultados, 'total': len(resultados)})


if __name__ == '__main__':
    logger.info("Servidor web iniciado", extra={'modulo': 'M8', 'funcao': 'main'})
    app.run(debug=True, host='0.0.0.0', port=5000)
