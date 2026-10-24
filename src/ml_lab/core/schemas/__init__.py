"""Esquemas Pydantic partilhados: configurações, hiperparâmetros, validação e serialização.

São os tipos que atravessam a fronteira do núcleo: a interface e a API constroem-nos,
os serviços validam-nos. Por isso a interface pode importá-los diretamente. A
serialização (``serialization.py``) faz o caminho inverso: passa o que os serviços
devolvem a JSON, para a API e para o frontend (ADR 0019).
"""
