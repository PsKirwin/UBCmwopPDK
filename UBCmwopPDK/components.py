"""Cells imported from the PDK."""

from functools import cache, partial

import gdsfactory as gf
import numpy as np
from gdsfactory import Component
from gdsfactory.typings import (
    Callable,
    ComponentReference,
    ComponentSpec,
    CrossSectionSpec,
    Label,
    LayerSpec,
    List,
    Optional,
    Port,
    Tuple,
)

from UBCmwopPDK import tech
from UBCmwopPDK.config import CONFIG
from UBCmwopPDK.import_gds import import_gc, import_gds
from UBCmwopPDK.tech import (
    LAYER,
    LAYER_STACK,
    add_pins_bbox_siepic,
)

um = 1e-6


@gf.cell
def bend_euler_sc(**kwargs) -> Component:
    kwargs.pop("cross_section", None)
    return gf.components.bend_euler(cross_section="xs_sc_devrec", **kwargs)


bend_euler180_sc = partial(bend_euler_sc, angle=180)
bend = bend_euler_sc


@gf.cell(post_process=(tech.add_pins_bbox_siepic,), include_module=True)
def straight(length: float = 1.0, npoints: int = 2, cross_section="xs_sc"):
    return gf.components.straight(
        length=length, npoints=npoints, cross_section=cross_section
    )


straight_heater_metal = partial(gf.c.straight_heater_metal, straight=straight)
bend_s = partial(
    gf.components.bend_s,
    cross_section="xs_sc",
)

info1550te = dict(polarization="te", wavelength=1.55)
info1310te = dict(polarization="te", wavelength=1.31)
info1550tm = dict(polarization="tm", wavelength=1.55)
info1310tm = dict(polarization="tm", wavelength=1.31)
thermal_phase_shifter_names = [
    "thermal_phase_shifter_multimode_500um",
    "thermal_phase_shifter_te_1310_500um",
    "thermal_phase_shifter_te_1310_500um_lowloss",
    "thermal_phase_shifter_te_1550_500um_lowloss",
]

prefix_te1550 = prefix_tm1550 = prefix_te1310 = prefix_tm1130 = "o2"


def clean_name(name: str) -> str:
    return name.replace("_", ".")


def thermal_phase_shifter0() -> gf.Component:
    """Return thermal_phase_shifters fixed cell."""
    return import_gds(
        "thermal_phase_shifters.gds", cellname=thermal_phase_shifter_names[0]
    )


def thermal_phase_shifter1() -> gf.Component:
    """Return thermal_phase_shifters fixed cell."""
    return import_gds(
        "thermal_phase_shifters.gds", cellname=thermal_phase_shifter_names[1]
    )


def thermal_phase_shifter2() -> gf.Component:
    """Return thermal_phase_shifters fixed cell."""
    return import_gds(
        "thermal_phase_shifters.gds", cellname=thermal_phase_shifter_names[2]
    )


def thermal_phase_shifter3() -> gf.Component:
    """Return thermal_phase_shifters fixed cell."""
    return import_gds(
        "thermal_phase_shifters.gds", cellname=thermal_phase_shifter_names[3]
    )


def ebeam_BondPad() -> gf.Component:
    """Return ebeam_BondPad fixed cell."""
    return import_gds("ebeam_BondPad.gds")


def ebeam_adiabatic_te1550() -> gf.Component:
    """Return ebeam_adiabatic_te1550 fixed cell."""
    return import_gds("ebeam_adiabatic_te1550.gds")


def ebeam_adiabatic_tm1550() -> gf.Component:
    """Return ebeam_adiabatic_tm1550 fixed cell."""
    return import_gds("ebeam_adiabatic_tm1550.gds")


def ebeam_bdc_te1550() -> gf.Component:
    """Return ebeam_bdc_te1550 fixed cell."""
    return import_gds("ebeam_bdc_te1550.gds")


def ebeam_bdc_tm1550() -> gf.Component:
    """Return ebeam_bdc_tm1550 fixed cell."""
    return import_gds("ebeam_bdc_tm1550.gds")


def ebeam_crossing4() -> gf.Component:
    """Return ebeam_crossing4 fixed cell."""
    return import_gds("ebeam_crossing4.gds")


@gf.cell
def straight_one_pin(length=1, cross_section=tech.strip_bbox) -> gf.Component:
    c = gf.Component()
    add_pins_left = partial(tech.add_pins_siepic, prefix="o1", pin_length=0.1)
    s = c << gf.components.straight(length=length, cross_section=cross_section)
    c.add_ports(s.ports)
    add_pins_left(c)
    c.absorb(s)
    return c


@gf.cell
def ebeam_crossing4_2ports() -> gf.Component:
    """Return ebeam_crossing4 fixed cell."""
    c = gf.Component()
    x = c << ebeam_crossing4()
    s1 = c << straight_one_pin()
    s2 = c << straight_one_pin()

    s1.connect("o1", x.ports["o2"])
    s2.connect("o1", x.ports["o4"])

    c.add_port(name="o1", port=x.ports["o1"])
    c.add_port(name="o4", port=x.ports["o3"])
    return c


def ebeam_splitter_adiabatic_swg_te1550() -> gf.Component:
    """Return ebeam_splitter_adiabatic_swg_te1550 fixed cell."""
    return import_gds("ebeam_splitter_adiabatic_swg_te1550.gds")


def ebeam_splitter_swg_assist_te1310() -> gf.Component:
    """Return ebeam_splitter_swg_assist_te1310 fixed cell."""
    return import_gds("ebeam_splitter_swg_assist_te1310.gds")


def ebeam_splitter_swg_assist_te1550() -> gf.Component:
    """Return ebeam_splitter_swg_assist_te1550 fixed cell."""
    return import_gds("ebeam_splitter_swg_assist_te1550.gds")


def ebeam_swg_edgecoupler() -> gf.Component:
    """Return ebeam_swg_edgecoupler fixed cell."""
    return import_gds("ebeam_swg_edgecoupler.gds")


def ebeam_terminator_te1310() -> gf.Component:
    """Return ebeam_terminator_te1310 fixed cell."""
    return import_gds("ebeam_terminator_te1310.gds")


def ebeam_terminator_te1550() -> gf.Component:
    """Return ebeam_terminator_te1550 fixed cell."""
    return import_gds("ebeam_terminator_te1550.gds")


def ebeam_terminator_tm1550() -> gf.Component:
    """Return ebeam_terminator_tm1550 fixed cell."""
    return import_gds("ebeam_terminator_tm1550.gds")


def ebeam_y_1550() -> gf.Component:
    """Return ebeam_y_1550 fixed cell."""
    return import_gds("ebeam_y_1550.gds")


def ebeam_y_adiabatic() -> gf.Component:
    """Return ebeam_y_adiabatic fixed cell."""
    return import_gds("ebeam_y_adiabatic.gds")


def ebeam_y_adiabatic_tapers() -> gf.Component:
    """Return ebeam_y_adiabatic fixed cell."""
    y = import_gds("ebeam_y_adiabatic.gds")
    return gf.add_tapers(y)


def ebeam_y_adiabatic_1310() -> gf.Component:
    """Return ebeam_y_adiabatic_1310 fixed cell."""
    return import_gds("ebeam_y_adiabatic_1310.gds")


def metal_via() -> gf.Component:
    """Return metal_via fixed cell."""
    return import_gds("metal_via.gds")


def photonic_wirebond_surfacetaper_1310() -> gf.Component:
    """Return photonic_wirebond_surfacetaper_1310 fixed cell."""
    return import_gds("photonic_wirebond_surfacetaper_1310.gds")


def photonic_wirebond_surfacetaper_1550() -> gf.Component:
    """Return photonic_wirebond_surfacetaper_1550 fixed cell."""
    return import_gds("photonic_wirebond_surfacetaper_1550.gds")


@gf.cell
def gc_te1310() -> gf.Component:
    """Return ebeam_gc_te1310 fixed cell."""
    c = gf.Component()
    gc = import_gc("ebeam_gc_te1310.gds")
    gc_ref = c << gc
    c.add_ports(gc_ref.ports)
    c.copy_child_info(gc)
    name = prefix_te1310
    c.add_port(
        name=name,
        port_type="vertical_te",
        center=(25, 0),
        layer=(1, 0),
        width=9,
    )
    c.info.update(info1310te)
    return c


@gf.cell
def gc_te1310_8deg() -> gf.Component:
    """Return ebeam_gc_te1310_8deg fixed cell."""
    c = gf.Component()
    gc = import_gc("ebeam_gc_te1310_8deg.gds")
    gc_ref = c << gc
    c.add_ports(gc_ref.ports)
    c.copy_child_info(gc)
    name = prefix_te1310
    c.add_port(
        name=name,
        port_type="vertical_te",
        center=(25, 0),
        layer=(1, 0),
        width=9,
    )
    c.info.update(info1310te)
    return c


@gf.cell
def gc_te1310_broadband() -> gf.Component:
    """Return ebeam_gc_te1310_broadband fixed cell."""
    c = gf.Component()
    gc = import_gc("ebeam_gc_te1310_broadband.gds")
    gc_ref = c << gc
    c.add_ports(gc_ref.ports)
    c.copy_child_info(gc)
    name = prefix_te1310
    c.add_port(
        name=name,
        port_type="vertical_te",
        center=(25, 0),
        layer=(1, 0),
        width=9,
    )
    c.info.update(info1310te)
    return c


@gf.cell
def gc_te1550() -> gf.Component:
    """Return ebeam_gc_te1550 fixed cell."""
    c = gf.Component()
    gc = import_gc("ebeam_gc_te1550.gds")
    gc_ref = c << gc
    c.add_ports(gc_ref.ports)
    c.copy_child_info(gc)
    name = prefix_te1550
    c.add_port(
        name=name,
        port_type="vertical_te",
        center=(25, 0),
        layer=(1, 0),
        width=9,
    )
    c.info.update(info1550te)
    return c


@gf.cell
def gc_te1550_90nmSlab() -> gf.Component:
    """Return ebeam_gc_te1550_90nmSlab fixed cell."""
    c = gf.Component()
    gc = import_gc("ebeam_gc_te1550_90nmSlab.gds")
    gc_ref = c << gc
    c.add_ports(gc_ref.ports)
    c.copy_child_info(gc)
    name = prefix_te1550
    c.add_port(
        name=name,
        port_type="vertical_te",
        center=(25, 0),
        layer=(1, 0),
        width=9,
    )
    c.info.update(info1550te)
    return c


@gf.cell
def gc_te1550_broadband() -> gf.Component:
    """Return ebeam_gc_te1550_broadband fixed cell."""
    c = gf.Component()
    gc = import_gc("ebeam_gc_te1550_broadband.gds")
    gc_ref = c << gc
    c.add_ports(gc_ref.ports)
    c.copy_child_info(gc)
    name = prefix_te1550
    c.add_port(
        name=name,
        port_type="vertical_te",
        center=(25, 0),
        layer=(1, 0),
        width=9,
    )
    c.info.update(info1550te)
    return c


@gf.cell
def gc_tm1550() -> gf.Component:
    """Return ebeam_gc_tm1550 fixed cell."""
    c = gf.Component()
    gc = import_gc("ebeam_gc_tm1550.gds")
    gc_ref = c << gc
    c.add_ports(gc_ref.ports)
    c.copy_child_info(gc)
    name = prefix_tm1550
    c.add_port(
        name=name,
        port_type="vertical_tm",
        center=(25, 0),
        layer=(1, 0),
        width=9,
    )
    c.info.update(info1550tm)
    return c


mzi = partial(
    gf.components.mzi,
    splitter=ebeam_y_1550,
    bend=bend_euler_sc,
    straight=straight,
    cross_section="xs_sc",
)

mzi_heater = partial(
    gf.components.mzi_phase_shifter,
    bend=bend_euler_sc,
    straight=straight,
    splitter=ebeam_y_1550,
)

via_stack_heater_mtop = partial(
    gf.components.via_stack,
    size=(10, 10),
    layers=(LAYER.M1_HEATER, LAYER.M2_ROUTER),
    vias=(None, None),
)


def get_input_label_text(
    port: Port,
    gc: ComponentReference,
    component_name: Optional[str] = None,
    username: str = CONFIG.username,
) -> str:
    """Return label for port and a grating coupler.

    Args:
        port: component port.
        gc: grating coupler reference.
        component_name: optional component name.
        username: for the label.
    """
    polarization = gc.info.get("polarization")
    wavelength = gc.info.get("wavelength")

    assert polarization.upper() in [
        "TE",
        "TM",
    ], f"Not valid polarization {polarization.upper()!r} in [TE, TM]"
    assert (
        isinstance(wavelength, int | float) and 1.0 < wavelength < 2.0
    ), f"{wavelength} is Not valid 1000 < wavelength < 2000"

    name = component_name or port.parent.metadata_child.get("name")
    name = clean_name(name)
    # return f"opt_{polarization.upper()}_{int(wavelength * 1000.0)}_device_{username}-{name}-{gc_index}-{port.name}"
    return f"opt_in_{polarization.upper()}_{int(wavelength * 1000.0)}_device_{username}-{name}"


def get_input_labels(
    io_gratings: List[ComponentReference],
    ordered_ports: List[Port],
    component_name: str,
    layer_label: Tuple[int, int] = (10, 0),
    gc_port_name: str = "o1",
    port_index: int = 1,
    get_input_label_text_function: Callable = get_input_label_text,
) -> List[Label]:
    """Return list of labels for all component ports.

    Args:
        io_gratings: list of grating_coupler references.
        ordered_ports: list of ports.
        component_name: name.
        layer_label: for the label.
        gc_port_name: grating_coupler port.
        port_index: index of the port.
        get_input_label_text_function: function.

    """
    gc = io_gratings[port_index]
    port = ordered_ports[1]

    text = get_input_label_text_function(
        port=port, gc=gc, component_name=component_name
    )
    layer, texttype = gf.get_layer(layer_label)
    label = Label(
        text=text,
        origin=gc.ports[gc_port_name].center,
        anchor="o",
        layer=layer,
        texttype=texttype,
    )
    return [label]


@gf.cell_with_child(include_module=True)
def add_fiber_array(
    component: ComponentSpec = straight,
    component_name: Optional[str] = None,
    gc_port_name: str = "o1",
    get_input_labels_function: Callable = get_input_labels,
    with_loopback: bool = False,
    optical_routing_type: int = 0,
    fanout_length: float = 0.0,
    grating_coupler: ComponentSpec = gc_te1550,
    cross_section: CrossSectionSpec = "xs_sc",
    layer_label: LayerSpec = LAYER.TEXT,
    straight: ComponentSpec = straight,
    **kwargs,
) -> Component:
    """Returns component with grating couplers and labels on each port.

    Routes all component ports south.
    Can add align_ports loopback reference structure on the edges.

    Args:
        component: to connect.
        component_name: for the label.
        gc_port_name: grating coupler input port name 'o1'.
        get_input_labels_function: function to get input labels for grating couplers.
        with_loopback: True, adds loopback structures.
        optical_routing_type: None: autoselection, 0: no extension.
        fanout_length: None  # if None, automatic calculation of fanout length.
        grating_coupler: grating coupler instance, function or list of functions.
        cross_section: spec.
        layer_label: for label.
        straight: straight component.

    """
    c = gf.Component()

    component = gf.routing.add_fiber_array(
        straight=straight,
        bend=bend,
        component=component,
        component_name=component_name,
        grating_coupler=grating_coupler,
        gc_port_name=gc_port_name,
        get_input_labels_function=get_input_labels_function,
        get_input_label_text_function=get_input_label_text,
        with_loopback=with_loopback,
        optical_routing_type=optical_routing_type,
        layer_label=layer_label,
        fanout_length=fanout_length,
        cross_section=cross_section,
        **kwargs,
    )
    ref = c << component
    ref.rotate(-90)
    c.add_ports(ref.ports)
    c.copy_child_info(component)
    return c


L = 1.55 / 4 / 2 / 2.44


@gf.cell
def dbg(
    w0: float = 0.5,
    dw: float = 0.1,
    n: int = 100,
    l1: float = L,
    l2: float = L,
) -> gf.Component:
    """Includes two ports.

    Args:
        w0: width.
        dw: delta width.
        n: number of elements.
        l1: length teeth1.
        l2: length teeth2.
    """
    c = gf.Component()
    s = gf.components.straight(length=l1, cross_section=tech.strip_simple)
    g = c << gf.components.dbr(
        w1=w0 - dw / 2,
        w2=w0 + dw / 2,
        n=n,
        l1=l1,
        l2=l2,
        cross_section=tech.strip_simple,
    )
    s1 = c << s
    s2 = c << s
    s1.connect("o2", g.ports["o1"])
    s2.connect("o2", g.ports["o2"])

    c.add_port("o1", port=s1.ports["o1"])
    c.add_port("o2", port=s2.ports["o1"])
    c = add_pins_bbox_siepic(c)
    return c


@gf.cell
def terminator_short(**kwargs) -> gf.Component:
    c = gf.Component()
    s = gf.components.taper(**kwargs, cross_section=tech.strip_simple)
    s1 = c << s
    c.add_port("o1", port=s1.ports["o1"])
    c = add_pins_bbox_siepic(c)
    return c


@gf.cell
def dbr(
    w0: float = 0.5,
    dw: float = 0.1,
    n: int = 100,
    l1: float = L,
    l2: float = L,
    cross_section: CrossSectionSpec = tech.strip_simple,
    **kwargs,
) -> gf.Component:
    """Returns distributed bragg reflector.

    Args:
        w0: width.
        dw: delta width.
        n: number of elements.
        l1: length teeth1.
        l2: length teeth2.
        cross_section: spec.
        kwargs: cross_section settings.
    """
    c = gf.Component()

    xs = gf.get_cross_section(cross_section, **kwargs)

    # add_pins_left = partial(add_pins_siepic, prefix="o1")
    s = c << gf.components.straight(length=l1, cross_section=xs)
    _dbr = gf.components.dbr(
        w1=w0 - dw / 2,
        w2=w0 + dw / 2,
        n=n,
        l1=l1,
        l2=l2,
        cross_section=xs,
    )
    dbr = c << _dbr
    s.connect("o2", dbr.ports["o1"])
    c.add_port("o1", port=s.ports["o1"])
    return add_pins_bbox_siepic(c)


@gf.cell(post_process=(tech.add_pins_bbox_siepic,), include_module=True)
def coupler(**kwargs) -> gf.Component:
    return gf.components.coupler(**kwargs).flatten()


@gf.cell(post_process=(tech.add_pins_bbox_siepic,), include_module=True)
def coupler_ring(**kwargs) -> gf.Component:
    return gf.components.coupler_ring(**kwargs).flatten()


@gf.cell(post_process=(tech.add_pins_bbox_siepic,), include_module=True)
def mmi1x2(**kwargs) -> gf.Component:
    return gf.components.mmi1x2(**kwargs)


@cache
def dbr_cavity(dbr=dbr, coupler=coupler, **kwargs) -> gf.Component:
    dbr = dbr(**kwargs)
    return gf.components.cavity(component=dbr, coupler=coupler)


@cache
def dbr_cavity_te(component="dbr_cavity", **kwargs) -> gf.Component:
    component = gf.get_component(component, **kwargs)
    return add_fiber_array(component=component)


spiral = partial(gf.components.spiral_external_io, cross_section=tech.xs_sc_devrec)

ebeam_dc_halfring_straight = coupler_ring


@gf.cell
def ebeam_dc_halfring_straight(
    gap: float = 0.2,
    radius: float = 5.0,
    length_x: float = 4.0,
    siepic: bool = True,
    model: str = "ebeam_dc_halfring_straight",
    **kwargs,
) -> gf.Component:
    r"""Return a ring coupler.

    Args:
        gap: spacing between parallel coupled straight waveguides.
        radius: of the bends.
        length_x: length of the parallel coupled straight waveguides.
        cross_section: cross_section spec.
        siepic: if True adds siepic.
        kwargs: cross_section settings for bend and coupler.

    .. code::

           2             3
           |             |
            \           /
             \         /
           ---=========---
         1    length_x    4


    """

    c = gf.Component()
    ref = c << coupler_ring(gap=gap, radius=radius, length_x=length_x, **kwargs)
    thickness = LAYER_STACK.get_layer_to_thickness()
    c.add_ports(ref.ports)

    if siepic:
        x = tech.xs_sc_simple
        c.info["model"] = model
        c.info["gap"] = gap
        c.info["radius"] = radius
        c.info["wg_thickness"] = thickness[LAYER.WG]
        c.info["wg_width"] = x.width
        c.info["Lc"] = length_x

    return c


@gf.cell
def supercon_CPW_resonator_IDC(
    coupler_spec: dict = None,
    length: float = 5000,  # length in [um]
    cross_section: CrossSectionSpec = "xs_supercon_CPW",
    cap_cross_section: CrossSectionSpec = "xs_supercon_CPW_cap",
    label: str = None,
    trace_layer=LAYER.SC_TRACE,
    gap_layer=LAYER.SC_GAP,
) -> gf.Component:
    """Returns a superconducting CPW resonator with an interdigital capacitor at the end.

    Args:
        coupler_spec: dict that contains parameters for the IDC coupler
        length: length of the resonator in um
        cross_section: CrossSectionSpec for the CPW
        cross_section: CrossSectionSpec for the CPW end cap (default is a open-circuit end, can change if you want a short circuit)
        label: drawn label to be patterned for identification. If left blank, no label will be drawn
        trace_layer: can change to arbitrary layer
        gap_layer: can change to arbitrary layer

    author: Phillip Kirwin (pkirwin@ece.ubc.ca)
    """
    cross_section = gf.get_cross_section(cross_section)

    if coupler_spec is None:
        coupler_spec = dict(
            fingers=10, finger_length=100, finger_gap=5, thickness=10, layer=(70, 0)
        )

    IDC = gf.components.interdigital_capacitor(
        fingers=coupler_spec["fingers"],
        finger_length=coupler_spec["finger_length"],
        finger_gap=coupler_spec["finger_gap"],
        thickness=coupler_spec["thickness"],
        layer=coupler_spec["layer"],
    )
    fingers = IDC.get_setting("fingers")
    finger_length = IDC.get_setting("finger_length")
    finger_gap = IDC.get_setting("finger_gap")
    thickness = IDC.get_setting("thickness")
    stub1_length = np.max([3 * thickness, 30])
    stub2_length = np.max([3 * thickness, 10])
    stub1 = gf.path.extrude(
        p=gf.path.straight(length=stub1_length),
        width=5 * cross_section.width,
        layer=trace_layer,
    )
    stub2 = gf.path.extrude(
        p=gf.path.straight(length=stub2_length),
        width=cross_section.width,
        layer=trace_layer,
    )

    IDCwithstubs = gf.Component("IDCwithstubs", with_uuid=False)
    IDC_ref = IDCwithstubs << IDC
    stub1_ref = IDCwithstubs << stub1
    stub2_ref = IDCwithstubs << stub2
    IDC_ref.connect("o1", stub1.ports["o2"])
    stub2_ref.connect("o1", IDC_ref.ports["o2"])

    xsize = finger_length + finger_gap + 2 * thickness + stub1_length + stub2_length
    ysize = fingers * thickness + (fingers - 1) * finger_gap + 20
    keepout_box = gf.components.rectangle(size=(xsize, ysize), layer=trace_layer)

    temp = gf.Component("temp")
    keepout_box_ref = temp << keepout_box
    keepout_box_ref.movey(-ysize / 2)
    keepout_gap = gf.geometry.boolean(
        B=IDCwithstubs,
        A=keepout_box_ref,
        operation="not",
        precision=1e-6,
        layer=gap_layer,
    )
    _keepout_gap_ref = IDCwithstubs << keepout_gap

    IDCwithstubs.add_port("o1", port=stub1_ref.ports["o1"])
    IDCwithstubs.add_port("o2", port=stub2_ref.ports["o2"])

    # resonator meander
    bend_radius = 100
    extra_length = 150  # additional length added to start of meander
    remainder_length = (
        length - (2 * np.pi * bend_radius + extra_length)
    ) / 3  # length of each of the straight sections
    res_path = gf.Path()
    left_turn = gf.path.arc(radius=bend_radius, angle=90)
    right_turn = gf.path.arc(radius=bend_radius, angle=-90)
    straight = gf.path.straight(length=remainder_length)  # 1322.22721173
    straight2 = gf.path.straight(length=extra_length)
    res_path.append(
        [
            straight2,
            straight,
            right_turn,
            right_turn,
            straight,
            left_turn,
            left_turn,
            straight,
        ]
    )
    res = gf.path.extrude(p=res_path, cross_section=cross_section)
    print(res_path.length())

    # resonator cap (to make it an open-circuit at the end)
    cap = gf.path.extrude(
        p=gf.path.straight(length=10), cross_section=cap_cross_section
    )

    # text label
    if label is not None:
        cellname = label + "_" + str(int(length)) + "um"
        label = gf.components.text(text=cellname, size=30, layer=gap_layer)
    else:
        cellname = "Res_" + str(int(length)) + "um"
        label = gf.components.text(text="", size=30, layer=gap_layer)
    # putting in parent cell
    c = gf.Component(cellname)
    IDCwithstubs_ref = c << IDCwithstubs
    res_ref = c << res
    cap_ref = c << cap
    label_ref = c << label
    label_ref.rotate(90).movex(6 * (finger_length + stub1_length)).movey(2 * ysize)
    res_ref.connect("e1", IDCwithstubs_ref.ports["o2"])
    cap_ref.connect("e1", res_ref.ports["e2"])

    c.add_port("e1", port=IDCwithstubs_ref.ports["o1"])

    return c


@gf.cell
def supercon_wire_resonator_IDC(
    coupler_spec: dict = None,
    lengths: float = [700, 1000, 1000, 500],  # length in [um]
    bend: gf.Path = gf.path.euler,
    bend_radius: float = 100,  # [um]
    wire_path: gf.Path = None,
    cross_section: CrossSectionSpec = "xs_supercon_wire",
    label: str = None,
    trace_layer=LAYER.SC_TRACE,
    gap_layer=LAYER.SC_GAP,
    offset: float = 100,
    pass_cross_section_to_bend: bool = True,
) -> gf.Component:
    """Returns a superconducting wire resonator with an interdigital capacitor at the end.
    Note that a ground plane will need to be defined separately. wire_path can be used
    to specify an arbitrary path for the wire to take. Otherwise, the wire will take
    the default path, which was originally intended for wrapping around an optical racetrack.

    Args:
        coupler_spec: dict that contains parameters for the IDC coupler
        lengths: array of side lengths of the resonator in um
        bend: bend spec. Path object.
        bend_radius: bend radius
        wire_path: arbitrary path for the resonator to take.
        cross_section: CrossSectionSpec for the wire
        label: drawn label to be patterned for identification. If left blank, no label will be drawn
        trace_layer: can change to arbitrary layer
        gap_layer: can change to arbitrary layer
        offset: offset between the wire and ground plane
        pass_cross_section_to_bend: pass cross_section to bend. defaults to True

    author: Phillip Kirwin (pkirwin@ece.ubc.ca)
    """
    xs = gf.get_cross_section(cross_section)
    bend_radius = bend_radius or xs.radius
    cross_section = xs.copy(radius=bend_radius)

    if coupler_spec is None:
        coupler_spec = dict(
            fingers=10,
            finger_length=100,
            finger_gap=5,
            thickness=10,
            layer=LAYER.SC_TRACE,
        )

    IDC = gf.components.interdigital_capacitor(
        fingers=coupler_spec["fingers"],
        finger_length=coupler_spec["finger_length"],
        finger_gap=coupler_spec["finger_gap"],
        thickness=coupler_spec["thickness"],
        layer=coupler_spec["layer"],
    )
    fingers = IDC.get_setting("fingers")
    finger_length = IDC.get_setting("finger_length")
    finger_gap = IDC.get_setting("finger_gap")
    thickness = IDC.get_setting("thickness")
    stub1_length = np.max([3 * thickness, 30])
    stub2_length = np.max([3 * thickness, 10])
    stub1 = gf.path.extrude(
        p=gf.path.straight(length=stub1_length),
        width=5 * cross_section.width,
        layer=trace_layer,
    )
    stub2 = gf.path.extrude(
        p=gf.path.straight(length=stub2_length),
        width=cross_section.width,
        layer=trace_layer,
    )

    IDCwithstubs = gf.Component("IDCwithstubs", with_uuid=False)
    IDC_ref = IDCwithstubs << IDC
    stub1_ref = IDCwithstubs << stub1
    stub2_ref = IDCwithstubs << stub2
    IDC_ref.connect("o1", stub1.ports["o2"])
    stub2_ref.connect("o1", IDC_ref.ports["o2"])

    xsize = finger_length + finger_gap + 2 * thickness + stub1_length + stub2_length
    ysize = fingers * thickness + (fingers - 1) * finger_gap + 20
    keepout_box = gf.components.rectangle(size=(xsize, ysize), layer=trace_layer)

    temp = gf.Component("temp")
    keepout_box_ref = temp << keepout_box
    keepout_box_ref.movey(-ysize / 2)
    keepout_gap = gf.geometry.boolean(
        B=IDCwithstubs,
        A=keepout_box_ref,
        operation="not",
        precision=1e-6,
        layer=gap_layer,
    )
    _keepout_gap_ref = IDCwithstubs << keepout_gap

    IDCwithstubs.add_port("o1", port=stub1_ref.ports["o1"])
    IDCwithstubs.add_port("o2", port=stub2_ref.ports["o2"])

    # resonator path and extrude

    bend = (
        partial(
            bend,
            radius=cross_section.radius,
        )
        if pass_cross_section_to_bend
        else partial(bend, radius=bend_radius)
    )
    north_straight = gf.path.straight(length=lengths[0])
    east_straight = gf.path.straight(length=lengths[1])
    south_straight = gf.path.straight(length=lengths[2])
    west_straight = gf.path.straight(length=lengths[3])

    res_path = gf.Path()
    res_path.append(
        [
            bend(angle=90),
            north_straight,
            bend(angle=-90),
            east_straight,
            bend(angle=-90),
            south_straight,
            bend(angle=-90),
            west_straight,
        ]
    )
    res_path = wire_path or res_path
    res = gf.path.extrude(p=res_path, cross_section=cross_section)
    length_total = res_path.length()
    print(length_total)
    # text label
    if label is not None:
        cellname = label + "_" + str(int(length_total)) + "um"
        label = gf.components.text(text=cellname, size=30, layer=gap_layer)
    else:
        cellname = "Res_" + str(int(length_total)) + "um"
        label = gf.components.text(text="", size=30, layer=gap_layer)
    # putting in parent cell
    c = gf.Component(cellname)
    IDCwithstubs_ref = c << IDCwithstubs
    res_ref = c << res
    label_ref = c << label
    label_ref.rotate(90).movex(6 * (finger_length + stub1_length)).movey(2 * ysize)
    res_ref.connect("e1", IDCwithstubs_ref.ports["o2"])

    c.add_port("e1", port=IDCwithstubs_ref.ports["o1"])
    c.info["coupler_length"] = float(xsize)

    gap_box = c << gf.components.rectangle(
        size=(
            lengths[1] + 2 * bend_radius + 2 * offset + cross_section.width,
            lengths[0] + 2 * bend_radius + 2 * offset + cross_section.width,
        ),
        layer=LAYER.SC_GAP,
    )
    gap_box.movex(xsize).movey(-offset - cross_section.width / 2)
    return c


@gf.cell
def ring_single_mod_coupler(
    gap: float = 0.2,
    radius: float = 10.0,
    length_x: float = 4.0,
    length_y: float = 0.6,
    bend: Component = gf.components.bend_euler,
    bend_coupler: Component = None,
    length_coupler: float = 4.0,
    offset_coupler: float = 0.0,
    radius_coupler: float = 10,
    straight: Component = straight,
    cross_section: CrossSectionSpec = "xs_sc",
    pass_cross_section_to_bend: bool = True,
) -> gf.Component:
    """Returns a single-bus ring. Based on the ring_single cell from the generic
    PDK, but adds parameters for making the length and position of the coupler different
    from the x-side length of the ring.

    Args:
        gap: gap between for coupler.
        radius: for the bends in the ring.
        length_x: ring coupler length.
        length_y: vertical straight length.
        bend: 90 degrees bend spec.
        bend_coupler: optional bend for coupler
        length_coupler: straight length of ring
        offset_coupler: offset of coupler from the bottom left
        radius_coupler: optional coupler radius.
        straight: straight spec.
        cross_section: cross_section spec.
        pass_cross_section_to_bend: pass cross_section to bend.


    author: Phillip Kirwin (pkirwin@ece.ubc.ca)
    """
    gap = gf.snap.snap_to_grid2x(gap)

    xs = gf.get_cross_section(cross_section)
    radius = radius or xs.radius
    cross_section = xs.copy(radius=radius)

    bend_coupler = bend_coupler or bend
    radius_coupler = radius_coupler or radius
    cross_section_coupler = xs.copy(radius=radius_coupler)

    c = gf.Component()

    # coupler
    cs = straight(length=length_coupler, cross_section=cross_section)
    bc = (
        bend_coupler(cross_section=cross_section_coupler)
        if pass_cross_section_to_bend
        else bend_coupler(radius=radius_coupler)
    )

    cs_ref = cs.ref()
    if length_coupler > 0:
        c.add(cs_ref)
    bcl = c << bc
    bcr = c << bc
    bcl.connect(port="o1", destination=cs_ref.ports["o1"])
    bcr.connect(port="o2", destination=cs_ref.ports["o2"])

    # ring
    sy = straight(length=length_y, cross_section=cross_section)
    sx = straight(length=length_x, cross_section=cross_section)
    b = (
        bend(cross_section=cross_section)
        if pass_cross_section_to_bend
        else bend(radius=radius)
    )
    sl = sy.ref()
    sr = sy.ref()
    st = sx.ref()
    sb = sx.ref()

    if length_y > 0:
        c.add(sl)
        c.add(sr)

    if length_x > 0:
        c.add(st)
        c.add(sb)

    bul = c << b
    bll = c << b
    bur = c << b
    blr = c << b

    sb.movey(cs.info["width"] + gap)
    sb.movex(-offset_coupler)

    blr.connect(port="o1", destination=sb.ports["o2"])
    sr.connect(port="o1", destination=blr.ports["o2"])
    bur.connect(port="o1", destination=sr.ports["o2"])
    st.connect(port="o1", destination=bur.ports["o2"])
    bul.connect(port="o1", destination=st.ports["o2"])
    sl.connect(port="o1", destination=bul.ports["o2"])
    bll.connect(port="o1", destination=sl.ports["o2"])

    c.add_port("o1", port=bcl.ports["o2"])
    c.add_port("o2", port=bcr.ports["o1"])

    c.info["length"] = (
        2 * sy.info["length"] + 2 * sx.info["length"] + 4 * b.info["length"]
    )

    return c


@gf.cell
def microwave_optical_resonator_system(
    gap: float = 1.0,
    length_x: float = 500,
    length_y: float = 500,
    radius: float = 100,
    bend: gf.Path = partial(gf.path.euler, use_eff=True),
    length_mw: float = 2400,
    op_gap: float = 0.2,
    op_bend_coupler: Component = None,
    op_length_coupler: float = 4.0,
    op_offset_coupler: float = 0.0,
    op_radius_coupler: float = 10,
    op_straight: Component = straight,
    op_cross_section: CrossSectionSpec = "xs_sc",
    op_pass_cross_section_to_bend: bool = True,
    mw_coupler_spec: dict = None,
    mw_cross_section: CrossSectionSpec = "xs_supercon_wire",
    mw_label: str = None,
    mw_pass_cross_section_to_bend: bool = True,
) -> gf.Component:
    """Compound element for microwave-optical transduction. Wraps a optical racetrack
    with a superconducting wire resonator.

    Args:
        gap: gap between the edges of the MW wire and the optical waveguide
        length_x: straight x-length of both resonators
        length_y: straight y-length of both resonators
        radius: radius of the bends on the racetrack
        bend: 90 degrees bend spec for both resonators.
        length_mw: total length of the microwave resonator
        op_gap: gap of directional coupler for the ring.
        op_bend_coupler: optional bend for coupler
        op_length_coupler: straight length of ring
        op_offset_coupler: offset of coupler from the bottom left
        op_radius_coupler: optional coupler radius.
        op_straight: straight spec.
        op_cross_section: cross_section spec.
        op_pass_cross_section_to_bend: pass cross_section to bend.
        mw_coupler_spec: dict that contains parameters for the IDC coupler
        mw_cross_section: CrossSectionSpec for the supercon wire
        mw_label: drawn label to be patterned for identification. If left blank, no label will be drawn
        mw_pass_cross_section_to_bend: pass cross_section to bend. defaults to True

    author: pkirwin@ece.ubc.ca
    """
    gap = gf.snap.snap_to_grid2x(gap)

    mw_xs = gf.get_cross_section(mw_cross_section)
    op_xs = gf.get_cross_section(op_cross_section)
    mw_radius = radius + gap + 0.5 * mw_xs.width + 0.5 * op_xs.width

    mw_bend_path = partial(bend, radius=mw_radius)

    op_bend_path = partial(bend, radius=radius)
    op_bend = partial(op_bend_path().extrude, cross_section=op_cross_section)

    length_remainder = length_mw - 4 * mw_bend_path().length() - 2 * length_y - length_x

    c = gf.Component()
    op_res = c << ring_single_mod_coupler(
        gap=op_gap,
        radius=radius,
        length_x=length_x,
        length_y=length_y,
        bend=op_bend,
        bend_coupler=op_bend_coupler,
        length_coupler=op_length_coupler,
        offset_coupler=op_offset_coupler,
        radius_coupler=op_radius_coupler,
        straight=op_straight,
        cross_section=op_cross_section,
        pass_cross_section_to_bend=op_pass_cross_section_to_bend,
    )
    mw_res = c << supercon_wire_resonator_IDC(
        coupler_spec=mw_coupler_spec,
        lengths=[length_y, length_x, length_y, length_remainder],
        bend=mw_bend_path,
        bend_radius=mw_radius,
        cross_section=mw_cross_section,
        label=mw_label,
        pass_cross_section_to_bend=mw_pass_cross_section_to_bend,
    )

    op_res.movey(-op_gap - op_xs.width / 2 + mw_xs.width / 2 + gap)
    op_res.movex(
        mw_res.info["coupler_length"]
        + radius
        + mw_radius
        + op_xs.width / 2
        + mw_xs.width / 2
        + gap
    )

    c.add_port("e1", port=mw_res.ports["e1"])
    c.add_port("o1", port=op_res.ports["o1"])
    c.add_port("o2", port=op_res.ports["o2"])
    return c


ring_single = partial(
    gf.components.ring_single,
    coupler_ring=coupler_ring,
    cross_section=tech.xs_sc,
    bend=bend,
    straight=straight,
    pass_cross_section_to_bend=False,
)
ring_double = partial(
    gf.components.ring_double,
    coupler_ring=coupler_ring,
    cross_section=tech.xs_sc,
    straight=straight,
)
ring_double_heater = partial(
    gf.components.ring_double_heater,
    coupler_ring=coupler_ring,
    via_stack=via_stack_heater_mtop,
    cross_section=tech.xs_sc,
    straight=straight,
    length_y=0.2,
)
ring_single_heater = partial(
    gf.components.ring_single_heater,
    coupler_ring=coupler_ring,
    via_stack=via_stack_heater_mtop,
    cross_section=tech.xs_sc,
    straight=straight,
)


ebeam_dc_te1550 = partial(
    gf.components.coupler,
)
taper = partial(gf.components.taper)
spiral = partial(gf.components.spiral_external_io)
ring_with_crossing = partial(
    gf.components.ring_single_dut,
    component=ebeam_crossing4_2ports,
    coupler=coupler_ring,
    port_name="o4",
    bend=bend,
    cross_section="xs_sc",
    straight=straight,
)


pad = partial(
    gf.components.pad,
    size=(75, 75),
    layer=LAYER.M2_ROUTER,
    bbox_layers=(LAYER.PAD_OPEN,),
    bbox_offsets=(-1.8,),
)


@gf.cell
def pad_supercon(
    size: float = (400.0, 400.0),
    gap: float = 130,
    buffer: float = 245,
    cross_section: CrossSectionSpec = "xs_supercon_CPW_feedline",
) -> gf.Component:
    """Returns a rectangular pad with a taper on the right side for microwave/RF lines.

    Args:
        size: x, y size.
        gap: distance between the pad and the ground plane (not drawn)
        buffer: length of gap layer to draw on left side of pad
        cross_section: crossSectionSpec of CPW that bondpad tapers into
    """

    s0 = gf.Section(
        width=size[1],
        offset=0,
        layer=LAYER.SC_TRACE,
        port_names=("e1", "e2"),
        name="_default",
    )
    s1 = gf.Section(
        width=gap, offset=(size[1] + gap) / 2, layer=LAYER.SC_GAP, name="top"
    )
    s2 = gf.Section(
        width=gap, offset=-(size[1] + gap) / 2, layer=LAYER.SC_GAP, name="bot"
    )
    xsec_pad = gf.CrossSection(sections=[s0, s1, s2])

    xsec_feedline = gf.get_cross_section(cross_section)

    pad = gf.path.extrude(p=gf.path.straight(length=size[0]), cross_section=xsec_pad)
    taper = gf.components.taper_cross_section_linear(
        cross_section1=xsec_pad, cross_section2=xsec_feedline, length=size[0]
    )
    c = gf.Component()
    pad_ref = c << pad
    taper_ref = c << taper
    taper_ref.connect("e1", pad_ref.ports["e2"])
    c.add_port("e1", port=taper_ref.ports["e2"])
    c.add_polygon(
        points=[
            (0, -(gap + size[1] / 2)),
            (-buffer, -(gap + size[1] / 2)),
            (-buffer, gap + size[1] / 2),
            (0, gap + size[1] / 2),
        ],
        layer=LAYER.SC_GAP,
    )
    return c


def add_label_electrical(component: Component, text: str, port_name: str = "e2"):
    """Adds labels for electrical port.

    Returns same component so it needs to be used as a decorator.
    """
    if port_name not in component.ports:
        raise ValueError(f"No port {port_name!r} in {list(component.ports.keys())}")

    component.add_label(
        text=text, position=component.ports[port_name].center, layer=LAYER.TEXT
    )
    return component


pad_array = partial(gf.components.pad_array, pad=pad, spacing=(125, 125))
add_pads_rf = partial(
    gf.routing.add_electrical_pads_top,
    component="ring_single_heater",
    pad_array=pad_array,
)
add_pads_dc = partial(
    gf.routing.add_electrical_pads_top_dc,
    component="ring_single_heater",
    pad_array=pad_array,
)


@cache
def add_fiber_array_pads_rf(
    component: ComponentSpec = "ring_single_heater",
    username: str = CONFIG.username,
    orientation: float = 0,
    **kwargs,
) -> Component:
    """Returns fiber array with label and electrical pads.

    Args:
        component: to add fiber array and pads.
        username: for the label.
        orientation: for adding pads.
        kwargs: for add_fiber_array.
    """
    c0 = gf.get_component(component)
    # text = f"elec_{username}-{clean_name(c0.name)}_G"
    # add_label = partial(add_label_electrical, text=text)
    c1 = add_pads_rf(component=c0, orientation=orientation)
    return add_fiber_array(component=c1, **kwargs)


@cache
def add_pads(
    component: ComponentSpec = "ring_single_heater",
    username: str = CONFIG.username,
    **kwargs,
) -> Component:
    """Returns fiber array with label and electrical pads.

    Args:
        component: to add fiber array and pads.
        username: for the label.
        kwargs: for add_fiber_array.
    """
    c0 = gf.get_component(component)
    # text = f"elec_{username}-{clean_name(c0.name)}_G"
    # add_label = partial(add_label_electrical, text=text)
    return add_pads_rf(component=c0, **kwargs)


if __name__ == "__main__":
    c = straight_heater_metal()
    c.pprint_ports()
    # c.pprint_ports()
    # c = straight()
    # c = uc.ring_single_heater()
    # c = uc.add_fiber_array_pads_rf(c)

    # c = ring_double(length_y=10)
    # c = ring_with_crossing()
    # c = mmi1x2()
    c = add_fiber_array(straight_heater_metal)
    # c = coupler_ring()
    # c = dbr_cavity_te()
    # c = dbr_cavity()
    # c = ring_single(radius=12)
    # c = ring_double(radius=12, length_x=2, length_y=2)
    # c = bend_euler()
    # c = mzi()
    # c = spiral()
    # c = pad_array()
    # c = ring_double_heater()
    # c = ring_single_heater()
    # c = ebeam_y_1550()
    # c = ebeam_dc_halfring_straight()
    # c = ring_with_crossing()
    # c = ring_single()
    c.pprint_ports()
    c.show(show_ports=False)
