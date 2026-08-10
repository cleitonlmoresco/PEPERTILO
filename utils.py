"""
Módulo de utilitários compartilhados.
Contém sanitização de tipos, conversão numpy, etc.
"""

import numpy as np

def to_native(obj, depth=0):
    """
    Converte QUALQUER objeto numpy para tipo Python nativo.
    Útil para serialização JSON.
    """
    if depth > 100:
        return str(obj)

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.str_):
        return str(obj)
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return to_native(obj.item(), depth + 1)
        return [to_native(x, depth + 1) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {to_native(k, depth + 1): to_native(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(x, depth + 1) for x in obj]
    if isinstance(obj, tuple):
        return tuple(to_native(x, depth + 1) for x in obj)
    if hasattr(obj, 'item') and callable(obj.item):
        try:
            return to_native(obj.item(), depth + 1)
        except:
            pass
    return obj

def estimar_tamanho_objeto(obj):
    """Estima o tamanho em bytes de um objeto (aprox.)"""
    import sys
    return sys.getsizeof(obj)
