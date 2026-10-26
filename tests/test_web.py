"""Testes da interface web do PEPERTILO.

Foco nos pontos que já quebraram em produção:
- upload sem arquivo / extensão inválida
- correção manual com corpo inválido (antes dava 500)
- conexões em formato dict/tupla
- exportação CSV/Excel (handle aberto no Windows, BOM para Excel pt-BR)
"""
import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'web'))

import app as webapp  # noqa: E402


def _cliente():
    app = webapp.create_app()
    app.config.update(TESTING=True)
    return app.test_client()


class TestUpload(unittest.TestCase):
    def setUp(self):
        webapp.jobs.clear()
        self.client = _cliente()
        self._original = webapp.processar_job_assincrono
        webapp.processar_job_assincrono = lambda job_id: None

    def tearDown(self):
        webapp.processar_job_assincrono = self._original
        webapp.jobs.clear()

    def test_index_responde(self):
        resposta = self.client.get('/')
        self.assertEqual(resposta.status_code, 200)

    def test_upload_sem_arquivo(self):
        resposta = self.client.post('/upload', data={})
        self.assertEqual(resposta.status_code, 400)

    def test_upload_extensao_invalida(self):
        dados = {'arquivos': (io.BytesIO(b'conteudo'), 'nota.txt')}
        resposta = self.client.post('/upload', data=dados, content_type='multipart/form-data')
        self.assertEqual(resposta.status_code, 400)
        self.assertIn('Nenhum arquivo válido', resposta.get_json()['erro'])

    def test_status_job_inexistente(self):
        resposta = self.client.get('/status/naoexiste')
        self.assertEqual(resposta.status_code, 404)


class TestCorrecao(unittest.TestCase):
    def setUp(self):
        webapp.jobs.clear()
        self.client = _cliente()
        webapp.jobs['job1'] = {
            'id': 'job1',
            'status': 'concluido',
            'resultado': {
                'conexoes': [('A1', 'ECU', 'BR', '1.5')],
                'funcoes': {'A1': 'Alimentação'},
            },
        }

    def tearDown(self):
        webapp.jobs.clear()

    def test_corpo_invalido_retorna_400(self):
        resposta = self.client.post('/api/corrigir/job1', data='não é json',
                                    content_type='application/json')
        self.assertEqual(resposta.status_code, 400)

    def test_indice_nao_numerico_retorna_400(self):
        resposta = self.client.post('/api/corrigir/job1', json={'indice': 'x', 'campo': 'cor', 'valor': 'AZ'})
        self.assertEqual(resposta.status_code, 400)

    def test_campo_invalido_retorna_400(self):
        resposta = self.client.post('/api/corrigir/job1', json={'indice': 0, 'campo': 'foo', 'valor': 'x'})
        self.assertEqual(resposta.status_code, 400)

    def test_indice_fora_do_intervalo_retorna_400(self):
        resposta = self.client.post('/api/corrigir/job1', json={'indice': 99, 'campo': 'cor', 'valor': 'AZ'})
        self.assertEqual(resposta.status_code, 400)

    def test_job_nao_concluido_retorna_409(self):
        webapp.jobs['job1']['status'] = 'processando'
        resposta = self.client.post('/api/corrigir/job1', json={'indice': 0, 'campo': 'cor', 'valor': 'AZ'})
        self.assertEqual(resposta.status_code, 409)

    def test_correcao_valida_atualiza_conexao(self):
        resposta = self.client.post('/api/corrigir/job1', json={'indice': 0, 'campo': 'cor', 'valor': 'AZ'})
        self.assertEqual(resposta.status_code, 200)
        self.assertEqual(webapp.jobs['job1']['resultado']['conexoes'][0], ('A1', 'ECU', 'AZ', '1.5'))

    def test_api_conexoes_aceita_tupla_e_dict(self):
        webapp.jobs['job1']['resultado']['conexoes'] = [
            ('A1', 'ECU', 'BR', '1.5'),
            {'pino': 'A2', 'destino': 'SENSOR', 'cor': 'AZ'},
        ]
        resposta = self.client.get('/api/conexoes/job1')
        self.assertEqual(resposta.status_code, 200)
        dados = resposta.get_json()
        self.assertEqual(dados['total'], 2)
        self.assertEqual(dados['conexoes'][0]['pino'], 'A1')
        self.assertEqual(dados['conexoes'][1]['pino'], 'A2')
        self.assertEqual(dados['conexoes'][1]['bitola'], '')


class TestDownload(unittest.TestCase):
    def setUp(self):
        webapp.jobs.clear()
        self.client = _cliente()

    def tearDown(self):
        webapp.jobs.clear()

    def test_csv_de_funcoes_tem_bom_e_conteudo(self):
        webapp.jobs['job2'] = {
            'id': 'job2',
            'status': 'concluido',
            'nome_modulo': 'Bancada',
            'resultado': {'conexoes': [], 'funcoes': {'A1': 'Alimentação'}},
        }
        resposta = self.client.get('/download/job2')
        self.assertEqual(resposta.status_code, 200)
        self.assertTrue(resposta.data.startswith(b'\xef\xbb\xbf'), 'CSV deve começar com BOM UTF-8')
        self.assertIn('Alimentação', resposta.data.decode('utf-8-sig'))

    def test_excel_e_gerado_e_removido(self):
        webapp.jobs['job3'] = {
            'id': 'job3',
            'status': 'concluido',
            'nome_modulo': 'ECU',
            'resultado': {'conexoes': [('A1', 'ECU', 'BR', '1.5')], 'funcoes': {'A1': 'Alimentação'}},
        }
        resposta = self.client.get('/download/job3')
        self.assertEqual(resposta.status_code, 200)
        self.assertGreater(len(resposta.data), 1000)
        self.assertTrue(resposta.data.startswith(b'PK'), 'XLSX é um zip')


if __name__ == '__main__':
    unittest.main()
