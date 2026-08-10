document.addEventListener('DOMContentLoaded', () => {
    const areaDrop = document.getElementById('areaDrop');
    const inputArquivo = document.getElementById('arquivo');
    const infoArquivo = document.getElementById('infoArquivo');
    const nomeArquivo = document.getElementById('nomeArquivo');
    const btnRemoverArquivo = document.getElementById('btnRemoverArquivo');
    const formUpload = document.getElementById('formUpload');
    const cardsModo = document.querySelectorAll('.card-modo');

    if (!areaDrop || !inputArquivo) return;

    // Drag & Drop
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

    areaDrop.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            inputArquivo.files = files;
            atualizarExibicaoArquivo(files);
        }
    });

    inputArquivo.addEventListener('change', () => {
        if (inputArquivo.files.length > 0) {
            atualizarExibicaoArquivo(inputArquivo.files);
        }
    });

    btnRemoverArquivo?.addEventListener('click', (e) => {
        e.stopPropagation();
        e.preventDefault();
        inputArquivo.value = '';
        infoArquivo.classList.add('oculto');
        nomeArquivo.textContent = '';
    });

    function atualizarExibicaoArquivo(files) {
        const nomes = Array.from(files).map(f => f.name).join(', ');
        nomeArquivo.textContent = nomes || 'Nenhum arquivo';
        infoArquivo.classList.remove('oculto');
    }

    // Seleção visual dos modos
    cardsModo.forEach(card => {
        card.addEventListener('click', () => {
            cardsModo.forEach(c => c.classList.remove('ativo'));
            card.classList.add('ativo');
            const radio = card.querySelector('input[type="radio"]');
            if (radio) radio.checked = true;
        });
    });

    // Intercepta o envio do formulário
    formUpload.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (!inputArquivo.files.length) {
            alert('Por favor, selecione pelo menos um arquivo.');
            return;
        }

        const btnEnviar = document.getElementById('btnEnviar');
        btnEnviar.disabled = true;
        btnEnviar.innerHTML = '<span>Enviando...</span>';

        const formData = new FormData(formUpload);

        try {
            const response = await fetch(formUpload.action, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Erro HTTP: ${response.status}`);
            }

            const data = await response.json();
            if (data.job_id) {
                // Redireciona para a tela de carregamento
                window.location.href = `/carregamento/${data.job_id}`;
            } else {
                alert('Erro: job_id não retornado.');
                btnEnviar.disabled = false;
                btnEnviar.innerHTML = '<span>Iniciar Análise</span>';
            }
        } catch (err) {
            console.error('Erro no upload:', err);
            alert('Falha ao enviar o arquivo. Verifique o console.');
            btnEnviar.disabled = false;
            btnEnviar.innerHTML = '<span>Iniciar Análise</span>';
        }
    });
});
