document.addEventListener('DOMContentLoaded', () => {
    const tabelaDados = document.querySelector('.tabela-dados');
    if (!tabelaDados) return;

    // Adiciona campo de busca/filtro em tempo real para a tabela de resultados
    const container = tabelaDados.closest('.painel-tabela');
    if (container) {
        const divFiltro = document.createElement('div');
        divFiltro.className = 'filtro-tabela-container';
        divFiltro.style.marginBottom = '1rem';

        const inputFiltro = document.createElement('input');
        inputFiltro.type = 'text';
        inputFiltro.placeholder = 'Filtrar componentes ou pinos na tabela...';
        inputFiltro.className = 'input-texto';
        inputFiltro.style.maxWidth = '350px';

        divFiltro.appendChild(inputFiltro);
        container.insertBefore(divFiltro, tabelaDados.parentElement);

        inputFiltro.addEventListener('input', (e) => {
            const termo = e.target.value.toLowerCase().trim();
            const linhas = tabelaDados.querySelectorAll('tbody tr');

            linhas.forEach(linha => {
                const textoLinha = linha.textContent.toLowerCase();
                if (textoLinha.includes(termo)) {
                    linha.style.display = '';
                } else {
                    linha.style.display = 'none';
                }
            });
        });
    }

    // Destaque visual ao passar o mouse nas linhas
    const linhasTabela = tabelaDados.querySelectorAll('tbody tr');
    linhasTabela.forEach(linha => {
        linha.addEventListener('mouseenter', () => {
            linha.style.backgroundColor = 'rgba(59, 130, 246, 0.08)';
        });
        linha.addEventListener('mouseleave', () => {
            linha.style.backgroundColor = '';
        });
    });
});
