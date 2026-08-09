"""
Módulo 7 - Consolidação e Exportação da Planilha.
Recebe as conexões rastreadas (Módulo 5) e as funções de pinos (Módulo 6)
e gera um arquivo Excel com abas "Pinagem Completa" e "Modo Bancada".
"""

import pandas as pd
import re
from logger_erros import logger, monitorar, ErroExportacao, Severidade

REGEX_BANCADA = re.compile(r'(30|15|31|BAT|IGN|GND|CAN[_ ]?[HL]|K[- ]?LINE|LIN)', re.IGNORECASE)

COLUNAS = [
    'Conector',
    'Pino',
    'Função',
    'Cor',
    'Bitola',
    'Componente Destino',
    'Página',
    'Confiança (%)',
    'Observações'
]


def _calcular_confianca(df):
    confianca = []
    for _, row in df.iterrows():
        score = 100.0
        if not row['Cor']:
            score -= 25
        if not row['Bitola']:
            score -= 15
        if row['Função'] in ['Desconhecida', 'Não documentado', '']:
            score -= 30
        confianca.append(max(score, 0))
    return confianca


def _gerar_observacoes(df):
    observacoes = []
    for _, row in df.iterrows():
        obs = []
        if not row['Cor']:
            obs.append('Cor não identificada')
        if not row['Bitola']:
            obs.append('Bitola não identificada')
        if row['Função'] in ['Desconhecida', 'Não documentado', '']:
            obs.append('Função não documentada')
        observacoes.append('; '.join(obs) if obs else '')
    return observacoes


@monitorar(modulo='M7')
def consolidar_conexoes(conexoes, pin_func=None):
    """
    Converte a lista de conexões em um DataFrame enriquecido.

    Args:
        conexoes: list of tuples (pino, destino, cor, bitola)
        pin_func: dict {pino: funcao} do Módulo 6, opcional

    Returns:
        pd.DataFrame com as colunas padronizadas.
    """
    if not conexoes:
        logger.warning("Lista de conexões vazia", extra={'modulo': 'M7'})
        return pd.DataFrame(columns=COLUNAS)

    df = pd.DataFrame(conexoes, columns=['Pino', 'Destino', 'Cor', 'Bitola'])

    # Renomeia a coluna 'Destino' para 'Componente Destino'
    df.rename(columns={'Destino': 'Componente Destino'}, inplace=True)

    if pin_func:
        df['Função'] = df['Pino'].map(pin_func).fillna('Não documentado')
    else:
        df['Função'] = 'Desconhecida'

    df['Conector'] = ''
    df['Página'] = 1
    df['Confiança (%)'] = _calcular_confianca(df)
    df['Observações'] = _gerar_observacoes(df)

    # Garante que todas as colunas esperadas estejam presentes
    for col in COLUNAS:
        if col not in df.columns:
            df[col] = ''

    return df[COLUNAS]


@monitorar(modulo='M7')
def gerar_excel(df, caminho_saida='pinagem_ECU.xlsx'):
    """
    Gera um arquivo Excel com duas abas:
    1. Pinagem Completa - todas as conexões
    2. Modo Bancada - apenas pinos críticos
    """
    if df.empty:
        raise ErroExportacao("DataFrame vazio, nada para exportar", severidade=Severidade.MEDIA)

    mask = df['Função'].str.contains(REGEX_BANCADA, na=False) | df['Pino'].str.contains(REGEX_BANCADA, na=False)
    df_bancada = df[mask].copy()
    df_completa = df.sort_values(by=['Conector', 'Pino'], na_position='last')

    with pd.ExcelWriter(caminho_saida, engine='openpyxl') as writer:
        df_completa.to_excel(writer, sheet_name='Pinagem Completa', index=False)
        df_bancada.to_excel(writer, sheet_name='Modo Bancada', index=False)

        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            for col in ws.columns:
                max_len = max((len(str(c.value or '')) for c in col), default=0)
                ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 40)

    logger.info(f"Planilha gerada: {caminho_saida} ({len(df_completa)} conexões)", extra={'modulo': 'M7'})
    return caminho_saida


@monitorar(modulo='M7')
def processar_e_exportar(conexoes, caminho_datasheet=None, saida='pinagem_ECU.xlsx'):
    """
    Fluxo completo: recebe conexões, opcionalmente extrai funções do datasheet
    e gera o Excel.
    """
    pin_func = None
    if caminho_datasheet:
        from extracao_datasheet import extrair_datasheet
        try:
            pin_func = extrair_datasheet(caminho_datasheet)
        except Exception as e:
            logger.warning(f"Datasheet ignorado: {e}", extra={'modulo': 'M7'})

    df = consolidar_conexoes(conexoes, pin_func)
    return gerar_excel(df, saida)


if __name__ == '__main__':
    conexoes_teste = [
        ('A1', 'Relé Principal', 'BR/VD', '1.5 mm²'),
        ('A2', 'Alternador', 'PT', '2.5 mm²'),
        ('B1', 'Sensor de Pressão', 'WH/BK', '0.75 mm²'),
        ('D12', 'Aterramento', 'PT', '2.5 mm²'),
        ('E3', 'CAN_H', 'WH', '0.5 mm²'),
    ]
    df = consolidar_conexoes(conexoes_teste)
    gerar_excel(df, 'teste_pinagem.xlsx')
    print("Planilha de teste gerada: teste_pinagem.xlsx")