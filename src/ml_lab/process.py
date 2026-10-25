"""O processo do ML Lab: pedir ao sistema operativo que não o trave.

No Windows 11, um processo sem janela em primeiro plano (um servidor lançado de um terminal,
como o Streamlit ou a API) pode ficar em modo de poupança de energia (*EcoQoS*). Num
processador híbrido, com núcleos de desempenho e de eficiência, passa para os núcleos lentos
ou para uma frequência baixa. Num i7-13620H, o treino de uma rede pequena ficava cinco vezes
mais lento ao fim de um segundo (de 3,5 para 19,5 ms por época), e o resto do ML Lab também.

``run_at_full_speed`` pede ao Windows, só para este processo, que não lhe aplique essa
poupança. Usa uma API documentada (``SetProcessInformation`` com ``ProcessPowerThrottling``),
não muda nenhuma definição do sistema e o efeito acaba quando o processo termina. Noutros
sistemas operativos não faz nada.
"""

import ctypes
from ctypes import wintypes
from typing import Final

_PROCESS_POWER_THROTTLING: Final = 4
"""``ProcessPowerThrottling``, na enumeração ``PROCESS_INFORMATION_CLASS`` do Windows."""

_THROTTLING_VERSION: Final = 1
_EXECUTION_SPEED: Final = 0x1


class _ThrottlingState(ctypes.Structure):
    """A estrutura ``PROCESS_POWER_THROTTLING_STATE`` do Windows."""

    _fields_ = (
        ("version", wintypes.ULONG),
        ("control_mask", wintypes.ULONG),
        ("state_mask", wintypes.ULONG),
    )


def run_at_full_speed() -> bool:
    """Pede ao Windows que não aplique a poupança de energia a este processo.

    Returns:
        Se o pedido foi aceite (sempre ``False`` fora do Windows).
    """
    # O WinDLL só existe no Windows: noutros sistemas, não há nada a pedir.
    win_dll = getattr(ctypes, "WinDLL", None)
    if win_dll is None:
        return False
    kernel32 = win_dll("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetProcessInformation.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    )
    kernel32.SetProcessInformation.restype = wintypes.BOOL
    # A velocidade de execução passa a ser controlada por nós (control_mask) e fica sem
    # poupança (state_mask a 0): é o "alta qualidade de serviço" da documentação do Windows.
    state = _ThrottlingState(_THROTTLING_VERSION, _EXECUTION_SPEED, 0)
    accepted = kernel32.SetProcessInformation(
        kernel32.GetCurrentProcess(),
        _PROCESS_POWER_THROTTLING,
        ctypes.byref(state),
        ctypes.sizeof(state),
    )
    return bool(accepted)
