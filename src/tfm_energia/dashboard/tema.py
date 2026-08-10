"""Paleta y estilo de los gráficos del cuadro de mando.

La paleta no se elige por gusto: está **verificada** con el validador de
accesibilidad del sistema de diseño. Resultados de esa comprobación:

* Modo claro, 4 series en pares adyacentes (áreas apiladas, barras, líneas):
  pasa todas las comprobaciones. Peor par en visión con deficiencia cromática
  ΔE 9,1 (objetivo ≥ 8) y en visión normal ΔE 22,9 (mínimo ≥ 15).
* Modo oscuro, los mismos cuatro tonos re-escalonados para el fondo oscuro:
  pasa todo, contraste incluido.
* Tres series con todos los pares comparados (dispersión): pasa todo.

Un aviso que condiciona el diseño: en modo claro el aqua (2,74:1) y el amarillo
(2,11:1) quedan por debajo de 3:1 sobre el fondo. La regla de compensación exige
**etiquetas visibles o vista de tabla**, de modo que la identidad de la serie
nunca dependa solo del color. Por eso cada gráfico del cuadro de mando lleva su
tabla desplegable.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Paleta:
    """Colores de un modo de visualización."""

    fondo: str
    plano: str
    tinta: str
    tinta_secundaria: str
    tinta_apagada: str
    rejilla: str
    eje: str
    series: tuple[str, ...]
    # Estados: reservados, nunca se reutilizan como serie
    bueno: str = "#0ca30c"
    aviso: str = "#fab219"
    grave: str = "#ec835a"
    critico: str = "#d03b3b"

    def color(self, i: int) -> str:
        """Color de la serie i, en orden fijo y sin ciclar."""
        if i >= len(self.series):
            raise IndexError(
                f"Solo hay {len(self.series)} ranuras de color. Una serie adicional "
                "debe agruparse en 'Otros' o separarse en gráficos pequeños, "
                "nunca recibir un tono generado."
            )
        return self.series[i]


CLARA = Paleta(
    fondo="#fcfcfb",
    plano="#f9f9f7",
    tinta="#0b0b0b",
    tinta_secundaria="#52514e",
    tinta_apagada="#898781",
    rejilla="#e1e0d9",
    eje="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
)

OSCURA = Paleta(
    fondo="#1a1a19",
    plano="#0d0d0d",
    tinta="#ffffff",
    tinta_secundaria="#c3c2b7",
    tinta_apagada="#898781",
    rejilla="#2c2c2a",
    eje="#383835",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
)

TIPOGRAFIA = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def layout_base(p: Paleta, alto: int = 380, titulo: str | None = None) -> dict:
    """Composición común: rejilla discreta, sin marco y con tinta de texto.

    El texto usa siempre tinta neutra, nunca el color de la serie: el color lo
    lleva la marca, y la etiqueta que la acompaña se lee igual sin él.
    """
    ejes = {
        "showgrid": True,
        "gridcolor": p.rejilla,
        "gridwidth": 1,
        "zeroline": False,
        "linecolor": p.eje,
        "tickfont": {"color": p.tinta_apagada, "size": 11},
        "title": {"font": {"color": p.tinta_secundaria, "size": 12}},
    }
    return {
        "height": alto,
        "paper_bgcolor": p.fondo,
        "plot_bgcolor": p.fondo,
        "font": {"family": TIPOGRAFIA, "color": p.tinta_secundaria, "size": 12},
        "title": (
            {"text": titulo, "font": {"color": p.tinta, "size": 15}, "x": 0, "xanchor": "left"}
            if titulo else None
        ),
        "margin": {"l": 60, "r": 24, "t": 48 if titulo else 24, "b": 44},
        "xaxis": {**ejes, "showgrid": False},
        "yaxis": dict(ejes),
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"color": p.tinta_secundaria, "size": 11},
            "bgcolor": "rgba(0,0,0,0)",
        },
        "hoverlabel": {
            "bgcolor": p.fondo,
            "bordercolor": p.eje,
            "font": {"family": TIPOGRAFIA, "color": p.tinta, "size": 12},
        },
    }
