from typing import Callable, List, Optional, Tuple

import pp
from phidl import device_layout as pd
from phidl.device_layout import Label
from pp.add_labels import get_input_label
from pp.cell import cell
from pp.component import Component
from pp.port import Port
from pp.types import ComponentFactory, ComponentReference

from ubc.config import CONFIG
from ubc.tech import LAYER

from ubc.components.grating_couplers import gc_te1550
from ubc.components.straight import straight

L = 1.55 / 4 / 2 / 2.44


def get_optical_text(
    port: Port,
    gc: ComponentReference,
    gc_index: Optional[int] = None,
    component_name: Optional[str] = None,
) -> str:
    """Return label for a component port and a grating coupler.

    Args:
        port: component port.
        gc: grating coupler reference.
        component_name: optional component name.
    """
    polarization = gc.get_property("polarization")
    wavelength_nm = gc.get_property("wavelength")

    assert polarization.upper() in [
        "TE",
        "TM",
    ], f"Not valid polarization {polarization.upper()} in [TE, TM]"
    assert (
        isinstance(wavelength_nm, (int, float)) and 1000 < wavelength_nm < 2000
    ), f"{wavelength_nm} is Not valid 1000 < wavelength < 2000"

    if component_name:
        name = component_name
    elif type(port.parent) == pp.Component:
        name = port.parent.name
    else:
        name = port.parent.ref_cell.name

    name = name.replace("_", "-")
    label = (
        f"opt_in_{polarization.upper()}_{int(wavelength_nm)}_device_"
        + f"{CONFIG.username}_({name})-{gc_index}-{port.name}"
    )
    return label


def get_input_labels_all(
    io_gratings,
    ordered_ports,
    component_name,
    layer_label=LAYER.LABEL,
    gc_port_name: str = "W0",
):
    """Return labels (elements list) for all component ports."""
    elements = []
    for i, g in enumerate(io_gratings):
        label = get_input_label(
            port=ordered_ports[i],
            gc=g,
            gc_index=i,
            component_name=component_name,
            layer_label=layer_label,
            gc_port_name=gc_port_name,
        )
        elements += [label]

    return elements


def get_input_labels(
    io_gratings: List[ComponentReference],
    ordered_ports: List[Port],
    component_name: str,
    layer_label: Tuple[int, int] = LAYER.LABEL,
    gc_port_name: str = "W0",
    port_index: int = 1,
) -> List[Label]:
    """Return labels (elements list) for all component ports."""
    if port_index == -1:
        return get_input_labels_all(
            io_gratings=io_gratings,
            ordered_ports=ordered_ports,
            component_name=component_name,
            layer_label=layer_label,
            gc_port_name=gc_port_name,
        )
    gc = io_gratings[port_index]
    port = ordered_ports[1]

    text = get_optical_text(
        port=port, gc=gc, gc_index=port_index, component_name=component_name
    )
    layer, texttype = pd._parse_layer(layer_label)
    label = pd.Label(
        text=text,
        position=gc.ports[gc_port_name].midpoint,
        anchor="o",
        layer=layer,
        texttype=texttype,
    )
    return [label]


@cell
def add_fiber_array(
    component: Component = straight,
    component_name: None = None,
    gc_port_name: str = "W0",
    get_input_labels_function: Callable = get_input_labels,
    with_loopback: bool = False,
    optical_routing_type: int = 0,
    fanout_length: float = 0.0,
    grating_coupler: ComponentFactory = gc_te1550,
    **kwargs,
) -> Component:
    """Returns component with grating couplers and labels on each port.

    Routes all component ports south.
    Can add align_ports loopback reference structure on the edges.

    Args:
        component: to connect
        component_name: for the label
        gc_port_name: grating coupler input port name 'W0'
        get_input_labels_function: function to get input labels for grating couplers
        with_loopback: True, adds loopback structures
        optical_routing_type: None: autoselection, 0: no extension
        fanout_length: None  # if None, automatic calculation of fanout length
        taper_length: length of the taper
        grating_coupler: grating coupler instance, function or list of functions
        optical_io_spacing: SPACING_GC
    """

    c = pp.routing.add_fiber_array(
        component=component,
        component_name=component_name,
        grating_coupler=grating_coupler,
        gc_port_name=gc_port_name,
        get_input_labels_function=get_input_labels_function,
        with_loopback=with_loopback,
        optical_routing_type=optical_routing_type,
        layer_label=LAYER.LABEL,
        fanout_length=fanout_length,
        **kwargs,
    )
    c.rotate(-90)
    return c


if __name__ == "__main__":
    import ubc.components as pdk

    # c = straight_no_pins()
    # c = add_fiber_array(component=c)
    # c = gc_tm1550()
    # print(c.get_ports_array())
    # print(c.ports.keys())
    c = pdk.straight()
    c = add_fiber_array(component=c)
    c.pprint()
    c.show()