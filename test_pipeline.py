"""Mini-script de prueba asíncrono del pipeline de extracción.

Uso:
    python test_pipeline.py                        # casos reales con LLM_PROVIDER del .env
    python test_pipeline.py --provider anthropic   # openai | anthropic | gemini
    python test_pipeline.py --offline              # simula fallos y reintentos sin API key
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from itertools import count

from dotenv import load_dotenv
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from chain import get_model, process_text

TEXTO_LOG = """\
2026-09-29 14:02:11 ERROR uvicorn.error: Exception in ASGI application (FastAPI)
redis.exceptions.ConnectionError: Too many connections
sqlalchemy.exc.TimeoutError: QueuePool limit of size 5 overflow 10 reached,
connection timed out, timeout 30 (PostgreSQL). Usuarios en producción reciben 504."""

TEXTO_ARQUITECTURA = """\
El frontend en React consume una API GraphQL escrita en Node.js que corre en
contenedores Docker orquestados con Kubernetes sobre AWS. Los eventos se publican en
Kafka y un worker en Python los guarda en MongoDB. Queremos agregar métricas con Prometheus."""

# Prueba de estrés: texto ambiguo, sin tecnologías explícitas.
TEXTO_AMBIGUO = "El sistema anda medio raro últimamente, a veces tarda y otras no. No sé qué pasa."


def titulo(texto: str) -> None:
    print(f"\n{'=' * 72}\n{texto}\n{'=' * 72}")


async def ejecutar(nombre: str, texto: str, **kwargs) -> bool:
    try:
        resultado = await process_text(texto, **kwargs)
    except Exception as exc:
        print(f"\n--- {nombre}: FALLÓ ({type(exc).__name__}) ---")
        return False
    print(f"\n--- {nombre}: OK ---\n{resultado.model_dump_json(indent=2)}")
    return True


# --- Casos reales -----------------------------------------------------------------------


async def casos_reales(provider: str | None) -> None:
    titulo("1) Log de error y descripción de arquitectura (en paralelo con asyncio.gather)")
    await asyncio.gather(
        ejecutar("Log de error", TEXTO_LOG, provider=provider),
        ejecutar("Arquitectura", TEXTO_ARQUITECTURA, provider=provider),
    )

    titulo("2) Prueba de estrés: texto ambiguo")
    await ejecutar("Texto ambiguo", TEXTO_AMBIGUO, provider=provider)

    titulo("3) Prueba de estrés: límite de tokens muy bajo (fuerza respuestas cortadas)")
    modelo_corto = get_model(provider, max_tokens=15)
    await ejecutar("max_tokens=15", TEXTO_LOG, model=modelo_corto)


# --- Casos offline: modelo falso que devuelve respuestas defectuosas --------------------


class ModeloFalso(GenericFakeChatModel):
    """Chat model falso que soporta `with_structured_output` (vía tool calling)."""

    def bind_tools(self, tools: Sequence, **kwargs):
        return self


_ids = count(1)


def tool_call(args: dict, finish_reason: str = "stop") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "EntidadesTecnicas", "args": args, "id": f"call_{next(_ids)}"}],
        response_metadata={"finish_reason": finish_reason},
    )


RESPUESTA_VALIDA = {
    "tecnologias": ["FastAPI", "Redis", "PostgreSQL", "redis"],  # el duplicado se limpia
    "nivel_de_criticidad": "Alta",  # se normaliza a "alta"
    "resumen_tecnico": "API FastAPI con Redis saturado y pool de PostgreSQL agotado en producción.",
}


async def casos_offline() -> None:
    titulo("Offline A) Respuesta cortada -> JSON inválido -> respuesta válida (se recupera)")
    modelo = ModeloFalso(
        messages=iter(
            [
                # Intento 1: el modelo se quedó sin tokens a mitad del JSON.
                tool_call({"tecnologias": ["FastAPI"]}, finish_reason="length"),
                # Intento 2: JSON completo pero viola el contrato (lista vacía, enum inválido).
                tool_call(
                    {"tecnologias": [], "nivel_de_criticidad": "high", "resumen_tecnico": "x"}
                ),
                # Intento 3: respuesta correcta.
                tool_call(RESPUESTA_VALIDA),
            ]
        )
    )
    await ejecutar("Recuperación", TEXTO_LOG, model=modelo)

    titulo("Offline B) El modelo nunca devuelve algo válido (se agotan los reintentos)")
    modelo = ModeloFalso(
        messages=iter([AIMessage(content="No puedo responder en JSON, perdón.")] * 3)
    )
    await ejecutar("Sin recuperación", TEXTO_AMBIGUO, model=modelo)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba del pipeline de extracción")
    parser.add_argument("--provider", choices=["openai", "anthropic", "gemini"], default=None)
    parser.add_argument("--offline", action="store_true", help="Simula fallos sin llamar a la API")
    parser.add_argument("--debug", action="store_true", help="Muestra también los logs DEBUG")
    args = parser.parse_args()

    logging.getLogger("pipeline_extraccion").setLevel(logging.DEBUG if args.debug else logging.INFO)
    # Aviso interno del SDK de Gemini que no afecta al pipeline.
    logging.getLogger("google_genai.models").setLevel(logging.ERROR)

    if args.offline:
        await casos_offline()
    else:
        await casos_reales(args.provider)


if __name__ == "__main__":
    # Consola de Windows: asegura que los acentos y la "ñ" se impriman bien.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    load_dotenv()
    asyncio.run(main())
