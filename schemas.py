"""Contrato de salida del pipeline: el objeto que el LLM está obligado a devolver.

Las restricciones viven en validadores de Pydantic (y no solo en el JSON Schema) para que
se apliquen igual con cualquier proveedor: si el modelo devuelve algo que no cumple el
contrato, se lanza `ValidationError` y la cadena lo reintenta.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

MIN_RESUMEN = 20
MAX_RESUMEN = 400


class NivelCriticidad(str, Enum):
    """Enum y no string libre: el modelo no puede inventar valores como "high" o "ALTA"."""

    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class EntidadesTecnicas(BaseModel):
    """Entidades técnicas extraídas de un texto (log de error, descripción de arquitectura, etc.)."""

    model_config = ConfigDict(extra="forbid")

    tecnologias: list[str] = Field(
        ...,
        description=(
            "Tecnologías, lenguajes, frameworks, bases de datos o herramientas mencionadas "
            "explícitamente en el texto, con su nombre oficial (p. ej. 'PostgreSQL', no 'postgres'). "
            "Debe tener al menos un elemento."
        ),
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        ...,
        description=(
            "Gravedad del problema o riesgo descripto: 'alta' si afecta producción, datos o "
            "seguridad; 'media' si degrada el servicio sin caída; 'baja' si es informativo o menor."
        ),
    )
    resumen_tecnico: str = Field(
        ...,
        description="Resumen técnico de 1 o 2 oraciones en español sobre el contenido del texto.",
    )

    @field_validator("tecnologias")
    @classmethod
    def tecnologias_no_vacias_sin_duplicados(cls, v: list[str]) -> list[str]:
        limpias = [t.strip() for t in v if t and t.strip()]
        if not limpias:
            raise ValueError("la lista de tecnologías no puede estar vacía")
        # Quita duplicados ignorando mayúsculas y conserva el orden original.
        unicas: dict[str, str] = {}
        for t in limpias:
            unicas.setdefault(t.casefold(), t)
        return list(unicas.values())

    @field_validator("nivel_de_criticidad", mode="before")
    @classmethod
    def normalizar_criticidad(cls, v: object) -> object:
        # Tolera " Alta " pero no sinónimos: "high" sigue fallando la validación.
        return v.strip().lower() if isinstance(v, str) else v

    @field_validator("resumen_tecnico")
    @classmethod
    def resumen_con_contenido(cls, v: str) -> str:
        v = " ".join(v.split())
        if not MIN_RESUMEN <= len(v) <= MAX_RESUMEN:
            raise ValueError(
                f"el resumen debe tener entre {MIN_RESUMEN} y {MAX_RESUMEN} caracteres "
                f"(tiene {len(v)})"
            )
        return v
