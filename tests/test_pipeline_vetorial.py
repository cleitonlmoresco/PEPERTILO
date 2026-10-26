"""Teste de ponta a ponta do caminho vetorial e do fluxo web.

Diferente dos testes unitários, aqui um PDF é gerado de verdade, passa pelo
roteador e pelo upload da interface Flask, e o Excel é baixado. Os bugs que
motivaram este teste:

- pdfplumber devolve y0/y1 de baixo para cima (KeyError 'y0');
- linewidth 0 (hairline) era descartado;
- stroking_color em escala de cinza vem como escalar;
- caixas desenhadas como path de 4 linhas não entram em page.rects;
- rótulos com espaço eram quebrados em palavras;
- processar_manual() não aceitava limite de páginas.
"""
import io
import sys
import tempfile
import time
import unittest
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'web'))

from roteador import rotear_arquivo  # noqa: E402


def gerar_diagrama(destino, num_pinos=30, paginas=1):
    """ECU com N pinos, um fio hairline por pino e rótulo de destino."""
    doc = fitz.open()
    cores = ["BR", "AZ", "VM", "PT", "VD", "LR"]

    def texto_centralizado(page, texto, cx, baseline_y, tamanho=8):
        largura = fitz.get_text_length(texto, fontname="helv", fontsize=tamanho)
        page.insert_text(fitz.Point(cx - largura / 2, baseline_y), texto, fontsize=tamanho)

    for _ in range(paginas):
        page = doc.new_page(width=595, height=842)
        page.draw_rect(fitz.Rect(220, 50, 380, 710), color=(0, 0, 0), width=1.5)
        texto_centralizado(page, "ECU", 300, 45)
        for i in range(num_pinos):
            y = 70 + i * 21
            cor = cores[i % len(cores)]
            page.insert_text(fitz.Point(224, y + 3), f"A{i + 1}", fontsize=8)
            page.draw_line(fitz.Point(220, y), fitz.Point(110, y), color=(0, 0, 0), width=0)
            texto_centralizado(page, f"SENSOR {i + 1}", 92, y + 3)
            texto_centralizado(page, cor, 150, y - 2, tamanho=6)
            texto_centralizado(page, "1.5 mm²", 185, y + 8, tamanho=6)

    doc.save(destino)
    doc.close()


class TestPipelineVetorial(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.pdf = str(Path(cls._tmp.name) / 'diagrama.pdf')
        gerar_diagrama(cls.pdf)
        cls.roteado = rotear_arquivo(cls.pdf)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_classifica_como_vetorial_sem_ocr(self):
        self.assertEqual(self.roteado['tipo'], 'pdf_vetorial')

    def test_status_ok_e_todas_as_conexoes(self):
        resultado = self.roteado['resultado']
        self.assertEqual(resultado['status'], 'ok')
        self.assertEqual(resultado['num_conexoes'], 30)
        self.assertEqual(resultado['num_pinos'], 30)

    def test_conexao_tem_pino_destino_cor_e_bitola(self):
        conexoes = {c[0]: c for c in self.roteado['resultado']['conexoes']}
        self.assertEqual(conexoes['A1'][1], 'SENSOR 1')
        self.assertEqual(conexoes['A1'][2], 'BR')
        self.assertEqual(conexoes['A1'][3], '1.5 mm²')
        self.assertEqual(conexoes['A2'][2], 'AZ')
        self.assertEqual(conexoes['A30'][1], 'SENSOR 30')

    def test_limite_de_paginas_vale_para_vetorial(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf = str(Path(tmp) / 'duas_paginas.pdf')
            gerar_diagrama(pdf, num_pinos=30, paginas=2)
            resultado = rotear_arquivo(pdf, limite_paginas=1)['resultado']
            self.assertEqual(resultado['num_paginas'], 1)


class TestFluxoWebCompleto(unittest.TestCase):
    """Upload -> processamento assíncrono -> conexões -> Excel."""

    @classmethod
    def setUpClass(cls):
        import app as webapp
        cls.webapp = webapp
        cls._tmp = tempfile.TemporaryDirectory()
        cls.pdf = Path(cls._tmp.name) / 'diagrama.pdf'
        gerar_diagrama(str(cls.pdf))
        cls.pdf_bytes = cls.pdf.read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.webapp.jobs.clear()
        cls._tmp.cleanup()

    def setUp(self):
        self.webapp.jobs.clear()
        app = self.webapp.create_app()
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def _esperar(self, job_id, timeout=60):
        limite = time.time() + timeout
        while time.time() < limite:
            dados = self.client.get(f'/status/{job_id}').get_json()
            if dados.get('status') in ('concluido', 'erro'):
                return dados
            time.sleep(0.2)
        self.fail(f'Job {job_id} não terminou em {timeout}s')

    def test_fluxo_completo(self):
        resposta = self.client.post(
            '/upload',
            data={
                'arquivos': (io.BytesIO(self.pdf_bytes), 'diagrama.pdf'),
                'nome_modulo': 'Teste',
            },
            content_type='multipart/form-data',
        )
        self.assertEqual(resposta.status_code, 200, resposta.get_data(as_text=True))
        job_id = resposta.get_json()['job_id']

        status = self._esperar(job_id)
        self.assertEqual(status['status'], 'concluido', status)
        self.assertEqual(status['progresso'], 100)

        conexoes = self.client.get(f'/api/conexoes/{job_id}').get_json()
        self.assertEqual(conexoes['total'], 30)
        self.assertEqual(conexoes['conexoes'][0]['pino'], 'A1')
        self.assertEqual(conexoes['conexoes'][0]['funcao'], 'Desconhecida')

        excel = self.client.get(f'/download/{job_id}')
        self.assertEqual(excel.status_code, 200)
        self.assertTrue(excel.data.startswith(b'PK'), 'XLSX é um arquivo zip')

        # não deixa arquivos enviados para trás
        for caminho in self.webapp.jobs[job_id].get('caminhos', []):
            Path(caminho).unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main()
