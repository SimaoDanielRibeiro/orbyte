"""Levar o modelo para outras apps: a preparação e o modelo num só ficheiro ONNX (fase 11).

O ONNX (Open Neural Network Exchange) é um formato aberto para modelos: um grafo de operações
com os valores aprendidos lá dentro. Abre-se com o ONNX Runtime, em Python, C#, Java,
JavaScript ou C++, sem o laboratório, sem o scikit-learn e sem o PyTorch.

O ``model.onnx`` de um pacote leva a preparação à frente do modelo (``preparation.py``), o
modelo convertido (``models.py``, um conversor por modelo) e as saídas com nome
(``onnx_file.py``). Antes de entrar no pacote, é confirmado nas linhas de exemplo.

O ONNX, o skl2onnx e o ONNX Runtime só se usam aqui (``tests/architecture``).
"""

from ml_lab.core.export.onnx_file import (
    OnnxInput,
    OnnxSignature,
    export_onnx,
    model_onnx,
    onnx_signature,
    preparation_onnx,
    run_onnx,
)

__all__ = [
    "OnnxInput",
    "OnnxSignature",
    "export_onnx",
    "model_onnx",
    "onnx_signature",
    "preparation_onnx",
    "run_onnx",
]
