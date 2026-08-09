document.addEventListener('DOMContentLoaded', () => {
    const formBusca = document.getElementById('formBusca');
    const termoBusca = document.getElementById('termoBusca');
    const filtroTipo = document.getElementById('filtroTipo');

    if (!formBusca) return;

    // Foco automático no campo de busca ao carregar a página
    if (termoBusca) {
        termoBusca.focus();
    }

    // Validação antes de enviar a busca
    formBusca.addEventListener('submit', (e) => {
        const valor = termoBusca ? termoBusca.value.trim() : '';
        if (!valor) {
            e.preventDefault();
            alert('Por favor, digite um termo para realizar a busca.');
            termoBusca.focus();
        }
    });

    // Submete o formulário automaticamente se o usuário alterar o filtro
    if (filtroTipo) {
        filtroTipo.addEventListener('change', () => {
            if (termoBusca && termoBusca.value.trim() !== '') {
                formBusca.submit();
            }
        });
    }
});
