"""Testes dos módulos puros (sem OCR): utils, consolidação e classificação."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from consolidacao_exportacao import COLUNAS, consolidar_conexoes, gerar_excel  # noqa: E402
from utils import to_native  # noqa: E402
import classificador  # noqa: E402


class TestToNative(unittest.TestCase):
    def test_tipos_numpy(self):
        self.assertEqual(to_native(np.int64(3)), 3)
        self.assertIsInstance(to_native(np.int64(3)), int)
        self.assertEqual(to_native(np.float32(1.5)), 1.5)
        self.assertIs(to_native(np.bool_(True)), True)

    def test_estruturas_aninhadas(self):
        entrada = {'a': np.array([1, 2]), 'b': (np.int32(7),)}
        saida = to_native(entrada)
        self.assertEqual(saida['a'], [1, 2])
        self.assertEqual(saida['b'], (7,))
        self.assertIsInstance(saida['b'][0], int)

    def test_profundidade_limite(self):
        objeto = atual = {}
        for _ in range(120):
            atual['n'] = {}
            atual = atual['n']
        self.assertIsInstance(to_native(objeto), dict)


class TestConsolidacao(unittest.TestCase):
    def test_colunas_e_confianca(self):
        conexoes = [
            ('A1', 'ECU', 'BR', '1.5'),
            ('A2', 'SENSOR', '', ''),
        ]
        df = consolidar_conexoes(conexoes, {'A1': 'Alimentação'})
        self.assertEqual(list(df.columns), COLUNAS)
        self.assertEqual(df.loc[0, 'Função'], 'Alimentação')
        self.assertEqual(df.loc[1, 'Função'], 'Não documentado')
        self.assertEqual(df.loc[0, 'Confiança (%)'], 100.0)
        self.assertEqual(df.loc[1, 'Confiança (%)'], 55.0)
        self.assertIn('Cor não identificada', df.loc[1, 'Observações'])

    def test_lista_vazia(self):
        df = consolidar_conexoes([])
        self.assertTrue(df.empty)
        self.assertEqual(list(df.columns), COLUNAS)

    def test_gerar_excel(self):
        df = consolidar_conexoes([('A1', 'ECU', 'BR', '1.5')], {'A1': 'Alimentação'})
        with tempfile.TemporaryDirectory() as tmp:
            destino = Path(tmp) / 'pinagem.xlsx'
            gerar_excel(df, str(destino))
            self.assertTrue(destino.exists())
            self.assertGreater(destino.stat().st_size, 1000)
            abas = pd.read_excel(destino, sheet_name=None)
            self.assertIn('Pinagem Completa', abas)


class TestClassificacao(unittest.TestCase):
    def test_extensoes(self):
        self.assertEqual(classificador.get_tipo_arquivo('x.PDF'), 'pdf')
        self.assertEqual(classificador.get_tipo_arquivo('foto.JPG'), 'imagem')
        self.assertEqual(classificador.get_tipo_arquivo('arquivo.docx'), 'desconhecido')

    def test_arquivo_inexistente_levanta(self):
        with self.assertRaises(classificador.ErroPipeline):
            classificador.classificar_arquivo(str(ROOT / 'nao_existe.pdf'))


if __name__ == '__main__':
    unittest.main()
