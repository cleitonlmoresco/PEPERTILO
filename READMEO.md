
```
# PEPERTILO 1.0

Sistema Integrado de Leitura, Análise, Restauração de Circuitos e Geração de Documentação Técnica.

## 📁 Estrutura do Projeto


```

PEPERTILO1.0/
├── logger_erros.py                (M0 – Logging e tratamento de erros)
├── classificador.py               (M1 – Classificação de arquivos)
├── extracao_vetorial.py           (M2 – Extração de PDFs vetoriais)
├── restauracao_img.py             (M3 – Restauração de imagens)
├── deteccao_simbolos.py           (M4 – Detecção de símbolos)
├── grafo_rastreador.py            (M5 – Grafo e rastreamento BFS)
├── extracao_datasheet.py          (M6 – Extração de datasheets)
├── consolidacao_exportacao.py     (M7 – Geração de planilhas Excel)
├── modo_mpu.py                    (MPU – Modo microcontrolador)
├── roteador.py                    (Orquestrador central)
├── requirements.txt
├── .gitignore
├── README.md
└── web/
├── app.py                     (M8 – Servidor Flask)
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── resultado.html
│   ├── resultado_mpu.html
│   └── busca.html
└── static/
├── css/
│   └── estilo.css
└── js/
├── upload.js
├── visualizador.js
└── busca.js

```

## ⚙️ Instalação e Execução

1. Certifique-se de estar com o Python instalado.
2. Instale as dependências listadas no projeto:
   ```bash
   pip install -r requirements.txt

```

3. Inicie o servidor web da aplicação:
```bash
python web/app.py

```


4. Acesse pelo navegador através do endereço local fornecido pelo Flask (geralmente `http://127.0.0.1:5000`).

## 🛠️ Tecnologias Utilizadas

* Python / Flask
* Processamento de Imagem e Visão Computacional
* Manipulação de PDFs Vetoriais e Planilhas Excel
* HTML5, CSS3 e JavaScript Moderno

```

---

Com este último arquivo salvo e enviado ao GitHub, o repositório **PEPERTILO 1.0** estará completo e com toda a estrutura pronta na nuvem! 🚀 Parabéns por concluir essa etapa do projeto!

```
