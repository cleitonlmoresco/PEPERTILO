
document.addEventListener('DOMContentLoaded', () => {
    const areaDrop = document.getElementById('areaDrop');
    const inputArquivo = document.getElementById('arquivo');
    const infoArquivo = document.getElementById('infoArquivo');
    const nomeArquivo = document.getElementById('nomeArquivo');
    const btnRemoverArquivo = document.getElementById('btnRemoverArquivo');
    const formUpload = document.getElementById('formUpload');
    const cardsModo = document.querySelectorAll('.card-modo');

    if (!areaDrop || !inputArquivo) return;

    // Arrastar e Soltar (Drag & Drop)
    ['dragenter', 'dragover'].forEach(eventName => {
        areaDrop.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            areaDrop.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        areaDrop.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            areaDrop.classList.remove('dragover');
        }, false);
    });

    // Evento de soltar o arquivo
    areaDrop.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            inputArquivo.files = files;
            atualizarExibicaoArquivo(files[0]);
        }
    });

    // Evento de seleção normal pelo botão
    inputArquivo.addEventListener('change', () => {
        if (inputArquivo.files.length > 0) {
            atualizarExibicaoArquivo(inputArquivo.files[0]);
        }
    });

    // Remover arquivo selecionado
    if (btnRemoverArquivo) {
        btnRemoverArquivo.addEventListener('click', (e) => {
            e.stopPropagation();
            e.preventDefault();
            inputArquivo.value = '';
            infoArquivo.classList.add('oculto');
            nomeArquivo.textContent = '';
        });
    }

    function atualizarExibicaoArquivo(file) {
        if (nomeArquivo && infoArquivo) {
            nomeArquivo.textContent = file.name;
            infoArquivo.classList.remove('oculto');
        }
    }

    // Seleção visual do card de modo
    cardsModo.forEach(card => {
        card.addEventListener('click', () => {
            cardsModo.forEach(c => c.classList.remove('ativo'));
            card.classList.add('ativo');
            const radio = card.querySelector('input[type="radio"]');
            if (radio) radio.checked = true;
        });
    });

    // Validação e feedback no envio
    if (formUpload) {
        formUpload.addEventListener('submit', (e) => {
            if (!inputArquivo.files.length) {
                e.preventDefault();
                alert('Por favor, selecione um arquivo antes de iniciar a análise.');
                return;
            }
            const btnEnviar = document.getElementById('btnEnviar');
            if (btnEnviar) {
                btnEnviar.disabled = true;
                btnEnviar.innerHTML = '<span>Processando...</span>';
            }
        });
    }
});
