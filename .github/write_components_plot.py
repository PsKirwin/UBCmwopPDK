import inspect
from enum import Enum

from UBCmwopPDK import PDK
from UBCmwopPDK.config import PATH

filepath = PATH.repo / "docs" / "components_plot.rst"

skip = {
    "LIBRARY",
    "circuit_names",
    "component_factory",
    "component_names",
    "container_names",
    "component_names_test_ports",
    "component_names_skip_test",
    "component_names_skip_test_ports",
    "dataclasses",
    "library",
    "waveguide_template",
}

skip_plot: tuple[str, ...] = ("add_fiber_array_siepic",)
skip_settings: tuple[str, ...] = ("flatten", "safe_cell_names")

cells = PDK.cells


with open(filepath, "w+") as f:
    f.write(
        """

Here are the components available in the PDK


Cells
=============================
"""
    )

    for name in sorted(cells.keys()):
        if name in skip or name.startswith("_"):
            continue
        print(name)
        sig = inspect.signature(cells[name])

        # Check if function has required parameters (no default value)
        has_required_params = any(
            param.default == inspect.Parameter.empty
            for param in sig.parameters.values()
        )

        kwargs_list = []
        for p in sig.parameters:
            default = sig.parameters[p].default
            if p in skip_settings:
                continue
            # Handle enum types
            if isinstance(default, Enum):
                enum_class = type(default).__name__
                enum_value = default.name
                kwargs_list.append(f"{p}={enum_class}.{enum_value}")
            # Handle basic types
            elif isinstance(default, int | float | str | tuple):
                kwargs_list.append(f"{p}={repr(default)}")
        kwargs = ", ".join(kwargs_list)

        # Skip plotting if function has required params or is in skip_plot list
        if name in skip_plot or has_required_params:
            f.write(
                f"""

{name}
----------------------------------------------------

.. autofunction:: UBCmwopPDK.cells.{name}

"""
            )
        else:
            f.write(
                f"""

{name}
----------------------------------------------------

.. autofunction:: UBCmwopPDK.cells.{name}

.. plot::
  :include-source:

  from UBCmwopPDK import PDK, cells
  from UBCmwopPDK.tech import LayerMapUbc

  PDK.activate()

  c = cells.{name}({kwargs})
  c.plot()

"""
            )
