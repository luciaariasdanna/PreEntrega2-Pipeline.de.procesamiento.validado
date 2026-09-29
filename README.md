# Pre-entrega 2 · Pipeline de extracción de entidades técnicas

Pipeline en **LangChain (LCEL)** que recibe un texto sin procesar (un log de error, una
descripción de arquitectura, un reporte de incidente) y devuelve un **objeto validado con
Pydantic** con las tecnologías mencionadas, el nivel de criticidad y un resumen técnico.

Funciona con **OpenAI**, **Anthropic** o **Gemini** (se elige por variable de entorno), y
reintenta solo si el LLM devuelve un JSON mal formado, incompleto o cortado por límite de tokens.

## Estructura

```
.
├── schemas.py         # Contrato Pydantic: EntidadesTecnicas + enum NivelCriticidad
├── chain.py           # Modelo, ChatPromptTemplate, cadena LCEL, reintentos y process_text()
├── test_pipeline.py   # Mini-script de prueba asíncrono (casos reales y simulación offline)
├── requirements.txt
├── .env.example
└── README.md
```

## Cómo funciona

```
{texto} ──► ChatPromptTemplate ──► model.with_structured_output(EntidadesTecnicas) ──► validar_salida ──► EntidadesTecnicas
            └───────────────────────────── .with_retry(stop_after_attempt=3) ─────────────────────┘
```

| Componente | Dónde | Qué hace |
|---|---|---|
| **Esquema Pydantic** | `schemas.py` | `tecnologias: list[str]` (no vacía, sin duplicados), `nivel_de_criticidad: Enum[baja, media, alta]`, `resumen_tecnico: str` (entre 20 y 400 caracteres). `extra="forbid"` rechaza campos inventados. |
| **Prompt template** | `chain.py` → `prompt` | `ChatPromptTemplate` con dos variables: `{texto}` (la entrada) y `{format_instructions}` (reglas de formato, cargadas con `.partial()`). Sin f-strings. |
| **Cadena LCEL** | `chain.py` → `build_chain()` | `prompt \| model.with_structured_output(EntidadesTecnicas, include_raw=True) \| validar_salida` |
| **Resiliencia** | `chain.py` → `.with_retry()` | Hasta 3 intentos con espera exponencial con *jitter*, solo ante `SalidaIncompletaError`, `OutputParserException` o `ValidationError`. |
| **Ejecución async** | `chain.py` → `process_text()` | Ejecuta la cadena con `.ainvoke()` y registra cada intento, los fallos y el resultado. |

### Cómo se detecta el `finish_reason`

`with_structured_output(..., include_raw=True)` devuelve, además del objeto parseado, el
mensaje crudo del LLM. Antes de aceptar el objeto, `validar_salida` revisa:

1. **Respuesta cortada:** si el `finish_reason` es `length` (OpenAI), `max_tokens` (Anthropic)
   o `MAX_TOKENS` (Gemini), lanza `SalidaIncompletaError` aunque el JSON parezca válido.
2. **JSON mal formado o que no cumple el esquema:** relanza el `parsing_error`
   (`OutputParserException` o `ValidationError`).
3. **Sin estructura:** si el modelo contestó texto libre en vez del objeto, lanza `SalidaIncompletaError`.

Cualquiera de esas excepciones dispara `.with_retry()`, que vuelve a ejecutar **toda** la
cadena (prompt + modelo + validación).

Estos errores **no** se reintentan en la cadena:
- **Red y rate limit:** ya los reintenta el SDK de cada proveedor.
- **API key inválida:** reintentar no la arregla.

### Logs

Un `BaseCallbackHandler` (`LogReintentosHandler`) escucha los eventos de la cadena y registra
cada intento. Cada ejecución tiene un id (p. ej. `[78b253]`) para que los logs no se mezclen
cuando se procesan varios textos en paralelo.

```
INFO    [78b253] Procesando texto (294 caracteres) con ModeloFalso
INFO    [78b253] Intento 1/3: invocando al modelo...
WARNING [78b253] Intento 1/3 falló (SalidaIncompletaError): respuesta cortada por límite de tokens (finish_reason='length', tokens_salida=?)
WARNING [78b253] Reintento 2/3: invocando al modelo de nuevo...
WARNING [78b253] Intento 2/3 falló (ValidationError): tecnologias: Value error, la lista de tecnologías no puede estar vacía; nivel_de_criticidad: Input should be 'baja', 'media' or 'alta'; ...
WARNING [78b253] Reintento 3/3: invocando al modelo de nuevo...
INFO    [78b253] Extracción validada en 4.6 s: {"tecnologias":["FastAPI","Redis","PostgreSQL"],"nivel_de_criticidad":"alta",...}
```

## Instalación

Requiere **Python 3.12** o superior.

```bash
python -m venv .venv
```

Activar el entorno virtual:

- Windows (PowerShell): `.venv\Scripts\Activate.ps1`
- Linux / macOS: `source .venv/bin/activate`

```bash
pip install -r requirements.txt
```

Copiá `.env.example` como `.env` y completá la API key del proveedor que vayas a usar. Son las
mismas variables que en la Pre-entrega 1. El `.env` está en `.gitignore`, así que no se sube al repo.

| Variable | Descripción | Default |
|---|---|---|
| `LLM_PROVIDER` | `openai`, `anthropic` o `gemini` | `gemini` |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` | API key del proveedor elegido | — |
| `OPENAI_MODEL` / `ANTHROPIC_MODEL` / `GEMINI_MODEL` | Modelo a usar | `gpt-4o-mini` / `claude-sonnet-5-5` / `gemini-flash-lite-latest` |
| `LLM_MAX_TOKENS` | Máximo de tokens de salida | `1024` |

La temperatura se fija en `0` (OpenAI y Gemini): para extracción estructurada buscamos
determinismo, no creatividad. Los modelos Claude actuales no aceptan `temperature`.

## Uso

### Desde código

```python
import asyncio
from dotenv import load_dotenv
from chain import process_text

load_dotenv()
resultado = asyncio.run(process_text("FastAPI devuelve 504: el pool de PostgreSQL está agotado."))
print(resultado.model_dump_json(indent=2))
```

### Mini-script de prueba

```bash
python test_pipeline.py                        # proveedor de LLM_PROVIDER
python test_pipeline.py --provider gemini      # openai | anthropic | gemini
python test_pipeline.py --offline              # simula fallos y reintentos, sin API key
python test_pipeline.py --debug                # agrega logs DEBUG (finish_reason, tokens)
```

Casos reales:

1. **Log de error** y **descripción de arquitectura**, procesados en paralelo con `asyncio.gather`.
2. **Prueba de estrés, texto ambiguo:** un texto sin tecnologías explícitas.
3. **Prueba de estrés, límite de tokens:** `max_tokens=15` obliga al modelo a cortar la
   respuesta, así que se ve la detección del `finish_reason`, los 3 intentos y el error final.

Casos offline (`--offline`), con un chat model falso que devuelve respuestas defectuosas:

- **A:** intento 1 cortado por tokens, intento 2 que viola el esquema e intento 3 válido. El pipeline **se recupera**.
- **B:** el modelo nunca devuelve la estructura. Se agotan los reintentos y el error llega controlado.

## Ejemplo de salida

Entrada:

```text
2026-09-29 14:02:11 ERROR uvicorn.error: Exception in ASGI application (FastAPI)
redis.exceptions.ConnectionError: Too many connections
sqlalchemy.exc.TimeoutError: QueuePool limit of size 5 overflow 10 reached,
connection timed out, timeout 30 (PostgreSQL). Usuarios en producción reciben 504.
```

Salida validada (corrida real con `gemini-flash-lite-latest`):

```json
{
  "tecnologias": ["Uvicorn", "FastAPI", "Redis", "SQLAlchemy", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "Se registra una falla crítica en producción debido a saturación de conexiones en Redis y agotamiento del pool de conexiones en SQLAlchemy con PostgreSQL, generando respuestas HTTP 504."
}
```

## Prueba de estrés: resultados observados

Resultados de la corrida real con `gemini-flash-lite-latest`:

| Escenario | Resultado |
|---|---|
| Texto ambiguo (*"El sistema anda medio raro últimamente..."*) | **El modelo se recuperó sin inventar nada:** `tecnologias: ["sistema no especificado"]`, criticidad `baja`, y el resumen aclara que *"no es posible determinar la causa raíz"*. Si hubiera devuelto la lista vacía, el validador lanzaría `ValidationError` y se reintentaría. |
| `max_tokens=15` | Los 3 intentos terminaron con `finish_reason='MAX_TOKENS'` (11 tokens de salida). El pipeline **detectó el JSON incompleto antes de parsearlo**, reintentó 3 veces y cortó con `SalidaIncompletaError` controlado. |
| Descripción de arquitectura | Extrajo las 10 tecnologías mencionadas, con criticidad `baja` porque no hay incidente. |

Casos cubiertos por los validadores (se ven con `--offline`):

| Escenario | Comportamiento |
|---|---|
| Criticidad fuera del enum (`"high"`) | `ValidationError`, que dispara el reintento. `" Alta "` se normaliza a `"alta"`. |
| Tecnologías duplicadas (`"Redis"`, `"redis"`) | Se deduplican sin distinguir mayúsculas y se conserva la primera. |
| API key inválida o cuota agotada | Falla en el primer intento, sin reintentos inútiles. El SDK del proveedor ya reintenta los errores 429 y 503. |

## Nota sobre el plan gratuito de Gemini

La cuota gratuita es **por modelo y por día**. `gemini-flash-latest` permite solo 20 pedidos
diarios y se agota rápido (el error es `429 RESOURCE_EXHAUSTED`). Por eso el default es
`gemini-flash-lite-latest`, que tiene una cuota propia más amplia.
