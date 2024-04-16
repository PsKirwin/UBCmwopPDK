"""UBC Siepic Ebeam PDK from edx course."""

from gdsfactory.config import PATH as GPATH
from gdsfactory.get_factories import get_cells
from gdsfactory.pdk import Pdk

from UBCmwopPDK import components, data, tech
from UBCmwopPDK.config import CONFIG, PATH
from UBCmwopPDK.tech import LAYER, LAYER_STACK, LAYER_VIEWS, cross_sections

try:
    from gplugins.sax.models import get_models

    from UBCmwopPDK import models

    models = get_models(models)
except ImportError:
    print("gplugins[sax] not installed, no simulation models available.")
    models = {}


__version__ = "2.5.0"

__all__ = [
    "CONFIG",
    "data",
    "PATH",
    "components",
    "tech",
    "LAYER",
    "cells",
    "cross_sections",
    "PDK",
    "__version__",
]


cells = get_cells(components)
PDK = Pdk(
    name="UBCmwopPDK",
    cells=cells,
    cross_sections=cross_sections,
    models=models,
    layers=dict(LAYER),
    layer_stack=LAYER_STACK,
    layer_views=LAYER_VIEWS,
)

GPATH.sparameters = PATH.sparameters
GPATH.interconnect = PATH.interconnect_cml_path
PDK.activate()


if __name__ == "__main__":
    m = get_models(models)
    for model in m.keys():
        print(model)
