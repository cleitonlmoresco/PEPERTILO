# PEPERTILO

Sistema de leitura de diagramas elétricos (PDF vetorial, PDF escaneado, imagem ou
foto), rastreamento das conexões e geração de planilhas de pinagem.

## Requisitos

- Python 3.10+ (testado com 3.11)
- Tesseract OCR instalado e no `PATH` (usado só no caminho de imagem/PDF rasterizado)
- Windows, Linux ou macOS

## Instalação

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

No Windows, se o Tesseract não estiver no `PATH`, instale em
`C:\Program Files\Tesseract-OCR` ou aponte `TESSERACT_CMD` para o executável.

## Rodar a interface web

```bash
python web/app.py
```

Acesse http://127.0.0.1:5000.

Variáveis de ambiente aceitas:

| Variável | Padrão | Descrição |
|---|---|---|
| `HOST` | `127.0.0.1` | Interface de escuta. Use `0.0.0.0` só em rede confiável. |
| `PORT` | `5000` | Porta do servidor. |
| `FLASK_DEBUG` | desligado | `1`/`true` liga o modo debug (nunca use exposto na rede). |
| `SECRET_KEY` | aleatória | Chave de sessão do Flask (defina em produção). |
| `MAX_FILE_MB` | `25` | Tamanho máximo por arquivo enviado. |
| `MAX_TOTAL_MB` | `100` | Tamanho máximo do envio inteiro. |

Os arquivos enviados ficam em `web/uploads/` e os logs em `logs/`; ambos são
ignorados pelo Git.

## Rodar pela linha de comando

```bash
python roteador.py <arquivo> [limite_paginas]
```

Exemplo:

```bash
python roteador.py diagrama.pdf
python roteador.py diagrama.pdf 5
```

O roteador classifica o arquivo e escolhe automaticamente o caminho:

- `pdf_vetorial` → extração direta da camada vetorial (rápido, sem OCR)
- `pdf_rasterizado` → renderização em 300 DPI + OCR
- `imagem_limpa` / `foto_celular` → restauração + OCR
- `manual` → extração de tabelas de datasheet

## Testes

```bash
python -m unittest discover -s tests -v
```

A suíte cobre os módulos puros, a API Flask e um fluxo de ponta a ponta que gera
um diagrama vetorial de 30 pinos, processa pelo roteador, faz upload na
interface e baixa o Excel.

## Estrutura

| Módulo | Arquivo | Função |
|---|---|---|
| M0 | `logger_erros.py` | Logging e tratamento de erros |
| M1 | `classificador.py` | Classificação do arquivo de entrada |
| M2 | `extracao_vetorial.py` | Extração de linhas, textos, retângulos e emendas |
| M3 | `restauracao_img.py` | Restauração de imagens |
| M4 | `deteccao_simbolos.py` | OCR e detecção de símbolos |
| M5 | `grafo_rastreador.py` | Grafo topológico e rastreamento BFS |
| M6 | `extracao_datasheet.py` | Extração de tabelas de datasheet |
| M7 | `consolidacao_exportacao.py` | Geração de CSV/Excel |
| M8 | `web/app.py` | Interface web Flask |
| M10 | `extracao_programadores.py` | Extração para programadores |
| M11 | `extracao_carprog.py` | Extração específica CarProg |
| — | `roteador.py` | Orquestrador central |

## Observações

- PDFs vetoriais usam `pdfplumber`; o `fitz` (PyMuPDF) é usado só para
  renderização e fallback de emendas.
- `import fitz` ainda funciona, mas o PyMuPDF 2.x avisa para usar
  `import pymupdf`. O aviso é benigno.
- O caminho de OCR é bem mais lento (~15–25 s por página); por isso PDFs com
  camada vetorial são processados sem OCR.
