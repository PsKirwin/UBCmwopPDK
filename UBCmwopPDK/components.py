"""Cells imported from the PDK."""

from functools import cache, partial

import gdsfactory as gf
import numpy as np
from gdsfactory import Component, ComponentReference
from gdsfactory.typings import (
    ComponentSpec,
    CrossSectionSpec,
)

from UBCmwopPDK import tech
from UBCmwopPDK.config import CONFIG, PATH
from UBCmwopPDK.import_gds import import_gds
from UBCmwopPDK.tech import (
    LAYER,
    add_pins_bbox_siepic,
)

um = 1e-6


@gf.cell
def bend_euler(cross_section="strip", **kwargs) -> Component:
    return gf.components.bend_euler(cross_section=cross_section, **kwargs)


bend_euler180 = partial(bend_euler, angle=180)
bend = bend_euler
bend_s = gf.c.bend_s


def straight(
    length: float = 10.0,
    npoints: int = 2,
    cross_section: CrossSectionSpec = "strip",
    **kwargs,
) -> Component:
    """Returns a Straight waveguide.

    Args:
        length: straight length (um).
        npoints: number of points.
        cross_section: specification (CrossSection, string or dict).
        kwargs: additional cross_section arguments.

    .. code::

        o1 -------------- o2
                length
    """
    return gf.c.straight(
        length=length, npoints=npoints, cross_section=cross_section, **kwargs
    )


@gf.cell
def wire_corner(cross_section="metal_routing", **kwargs) -> Component:
    return gf.c.wire_corner(cross_section=cross_section, **kwargs)


@gf.cell
def straight_heater_metal(length: float = 320.0, cross_section="strip") -> gf.Component:
    c = gf.c.straight_heater_metal(length=length, cross_section=cross_section)
    return c


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
        PATH.gds / "thermal_phase_shifters.gds", cellname=thermal_phase_shifter_names[0]
    )


def thermal_phase_shifter1() -> gf.Component:
    """Return thermal_phase_shifters fixed cell."""
    return import_gds(
        PATH.gds / "thermal_phase_shifters.gds", cellname=thermal_phase_shifter_names[1]
    )


def thermal_phase_shifter2() -> gf.Component:
    """Return thermal_phase_shifters fixed cell."""
    return import_gds(
        PATH.gds / "thermal_phase_shifters.gds", cellname=thermal_phase_shifter_names[2]
    )


def thermal_phase_shifter3() -> gf.Component:
    """Return thermal_phase_shifters fixed cell."""
    return import_gds(
        PATH.gds / "thermal_phase_shifters.gds", cellname=thermal_phase_shifter_names[3]
    )


def ebeam_BondPad() -> gf.Component:
    """Return ebeam_BondPad fixed cell."""
    return import_gds(PATH.gds / "ebeam_BondPad.gds")


def ebeam_adiabatic_te1550() -> gf.Component:
    """Return ebeam_adiabatic_te1550 fixed cell."""
    return import_gds(PATH.gds / "ebeam_adiabatic_te1550.gds")


def ebeam_adiabatic_tm1550() -> gf.Component:
    """Return ebeam_adiabatic_tm1550 fixed cell."""
    return import_gds(PATH.gds / "ebeam_adiabatic_tm1550.gds")


def ebeam_bdc_te1550() -> gf.Component:
    """Return ebeam_bdc_te1550 fixed cell."""
    return import_gds(PATH.gds / "ebeam_bdc_te1550.gds")


def ebeam_bdc_tm1550() -> gf.Component:
    """Return ebeam_bdc_tm1550 fixed cell."""
    return import_gds(PATH.gds / "ebeam_bdc_tm1550.gds")


def ebeam_crossing4() -> gf.Component:
    """Return ebeam_crossing4 fixed cell."""
    return import_gds(PATH.gds / "ebeam_crossing4.gds")


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
    c.flatten()
    return c


@gf.cell
def ebeam_splitter_adiabatic_swg_te1550() -> gf.Component:
    """Return ebeam_splitter_adiabatic_swg_te1550 fixed cell."""
    return import_gds(PATH.gds / "ebeam_splitter_adiabatic_swg_te1550.gds")


@gf.cell
def ebeam_splitter_swg_assist_te1310() -> gf.Component:
    """Return ebeam_splitter_swg_assist_te1310 fixed cell."""
    return import_gds(PATH.gds / "ebeam_splitter_swg_assist_te1310.gds")


@gf.cell
def ebeam_splitter_swg_assist_te1550() -> gf.Component:
    """Return ebeam_splitter_swg_assist_te1550 fixed cell."""
    return import_gds(PATH.gds / "ebeam_splitter_swg_assist_te1550.gds")


@gf.cell
def ebeam_swg_edgecoupler() -> gf.Component:
    """Return ebeam_swg_edgecoupler fixed cell."""
    return import_gds(PATH.gds / "ebeam_swg_edgecoupler.gds")


@gf.cell
def ebeam_terminator_te1310() -> gf.Component:
    """Return ebeam_terminator_te1310 fixed cell."""
    return import_gds(PATH.gds / "ebeam_terminator_te1310.gds")


@gf.cell
def ebeam_terminator_te1550() -> gf.Component:
    """Return ebeam_terminator_te1550 fixed cell."""
    return import_gds(PATH.gds / "ebeam_terminator_te1550.gds")


@gf.cell
def ebeam_terminator_tm1550() -> gf.Component:
    """Return ebeam_terminator_tm1550 fixed cell."""
    return import_gds(PATH.gds / "ebeam_terminator_tm1550.gds")


def ebeam_y_1550() -> gf.Component:
    """Return ebeam_y_1550 fixed cell."""
    return import_gds(PATH.gds / "ebeam_y_1550.gds")


@gf.cell
def ebeam_y_adiabatic() -> gf.Component:
    """Return ebeam_y_adiabatic fixed cell."""
    return import_gds(PATH.gds / "ebeam_y_adiabatic.gds")


@gf.cell
def ebeam_y_adiabatic_1310() -> gf.Component:
    """Return ebeam_y_adiabatic_1310 fixed cell."""
    return import_gds(PATH.gds / "ebeam_y_adiabatic_1310.gds")


@gf.cell
def metal_via() -> gf.Component:
    """Return metal_via fixed cell."""
    return import_gds(PATH.gds / "metal_via.gds")


@gf.cell
def photonic_wirebond_surfacetaper_1310() -> gf.Component:
    """Return photonic_wirebond_surfacetaper_1310 fixed cell."""
    return import_gds(PATH.gds / "photonic_wirebond_surfacetaper_1310.gds")


@gf.cell
def photonic_wirebond_surfacetaper_1550() -> gf.Component:
    """Return photonic_wirebond_surfacetaper_1550 fixed cell."""
    return import_gds(PATH.gds / "photonic_wirebond_surfacetaper_1550.gds")


@gf.cell
def gc_te1310() -> gf.Component:
    """Return ebeam_gc_te1310 fixed cell."""
    c = gf.Component()
    gc = import_gds(PATH.gds / "ebeam_gc_te1310.gds")
    gc_ref = c << gc
    gc_ref.dmirror()
    c.add_ports(gc_ref.ports)
    c.copy_child_info(gc)
    name = prefix_te1310
    c.add_port(
        name=name,
        port_type="vertical_te",
        center=(25, 0),
        layer=(1, 0),
        width=9,
        orientation=0,
    )
    c.info.update(info1310te)
    c.flatten()
    return c


@gf.cell
def gc_te1310_8deg() -> gf.Component:
    """Return ebeam_gc_te1310_8deg fixed cell."""
    c = gf.Component()
    gc = import_gds(PATH.gds / "ebeam_gc_te1310_8deg.gds")
    gc_ref = c << gc
    gc_ref.dmirror()
    c.add_ports(gc_ref.ports)
    c.copy_child_info(gc)
    name = prefix_te1310
    c.add_port(
        name=name,
        port_type="vertical_te",
        center=(25, 0),
        layer=(1, 0),
        width=9,
        orientation=0,
    )
    c.info.update(info1310te)
    c.flatten()
    return c


@gf.cell
def gc_te1310_broadband() -> gf.Component:
    """Return ebeam_gc_te1310_broadband fixed cell."""
    c = gf.Component()
    gc = import_gds(PATH.gds / "ebeam_gc_te1310_broadband.gds")
    gc_ref = c << gc
    gc_ref.dmirror()
    c.add_ports(gc_ref.ports)
    c.copy_child_info(gc)
    name = prefix_te1310
    c.add_port(
        name=name,
        port_type="vertical_te",
        center=(25, 0),
        layer=(1, 0),
        width=9,
        orientation=0,
    )
    c.info.update(info1310te)
    c.flatten()
    return c


@gf.cell
def gc_te1550() -> gf.Component:
    """Return ebeam_gc_te1550 fixed cell."""
    c = gf.Component()
    gc = import_gds(PATH.gds / "ebeam_gc_te1550.gds")
    gc_ref = c << gc
    gc_ref.dmirror()
    c.add_ports(gc_ref.ports)
    c.copy_child_info(gc)
    name = prefix_te1550
    c.add_port(
        name=name,
        port_type="vertical_te",
        center=(25, 0),
        layer=(1, 0),
        width=9,
        orientation=0,
    )
    c.info.update(info1550te)
    c.flatten()
    return c


@gf.cell
def gc_te1550_90nmSlab() -> gf.Component:
    """Return ebeam_gc_te1550_90nmSlab fixed cell."""
    c = gf.Component()
    gc = import_gds(PATH.gds / "ebeam_gc_te1550_90nmSlab.gds")
    gc_ref = c << gc
    gc_ref.dmirror()
    c.add_ports(gc_ref.ports)
    c.copy_child_info(gc)
    name = prefix_te1550
    c.add_port(
        name=name,
        port_type="vertical_te",
        center=(25, 0),
        layer=(1, 0),
        width=9,
        orientation=0,
    )
    c.info.update(info1550te)
    c.flatten()
    return c


@gf.cell
def gc_te1550_broadband() -> gf.Component:
    """Return ebeam_gc_te1550_broadband fixed cell."""
    c = gf.Component()
    gc = import_gds(PATH.gds / "ebeam_gc_te1550_broadband.gds")
    gc_ref = c << gc
    gc_ref.dmirror()
    c.add_ports(gc_ref.ports)
    c.copy_child_info(gc)
    name = prefix_te1550
    c.add_port(
        name=name,
        port_type="vertical_te",
        center=(25, 0),
        layer=(1, 0),
        width=9,
        orientation=0,
    )
    c.info.update(info1550te)
    c.flatten()
    return c


@gf.cell
def gc_tm1550() -> gf.Component:
    """Return ebeam_gc_tm1550 fixed cell."""
    c = gf.Component()
    gc = import_gds(PATH.gds / "ebeam_gc_tm1550.gds")
    gc_ref = c << gc
    gc_ref.dmirror()
    c.add_ports(gc_ref.ports)
    c.copy_child_info(gc)
    name = prefix_tm1550
    c.add_port(
        name=name,
        port_type="vertical_tm",
        center=(25, 0),
        layer=(1, 0),
        width=9,
        orientation=0,
    )
    c.info.update(info1550tm)
    c.flatten()
    return c


mzi = partial(
    gf.components.mzi,
    splitter=ebeam_y_1550,
    bend=bend_euler,
    straight="straight",
    cross_section="strip",
)

_mzi_heater = partial(
    gf.components.mzi_phase_shifter,
    bend="bend_euler",
    straight="straight",
    splitter="ebeam_y_1550",
    straight_x_top="straight_heater_metal",
)


@gf.cell
def mzi_heater(delta_length=10.0, length_x=320) -> gf.Component:
    """Returns MZI with heater.

    Args:
        delta_length: extra length for mzi arms.
    """
    return _mzi_heater(delta_length=delta_length, length_x=length_x)


@gf.cell
def via_stack_heater_mtop(size=(10, 10)) -> gf.Component:
    return gf.components.via_stack(
        size=size,
        layers=(LAYER.M1_HEATER, LAYER.M2_ROUTER),
        vias=(None, None),
    )


def get_input_label_text(
    gc: ComponentReference,
    component_name: str | None = None,
    username: str = CONFIG.username,
) -> str:
    """Return label for port and a grating coupler.

    Args:
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

    name = component_name
    name = clean_name(name)
    # return f"opt_{polarization.upper()}_{int(wavelength * 1000.0)}_device_{username}-{name}-{gc_index}-{port.name}"
    return f"opt_in_{polarization.upper()}_{int(wavelength * 1000.0)}_device_{username}-{name}"


def add_fiber_array(
    component: ComponentSpec = straight,
    component_name: str | None = None,
    gc_port_name: str = "o1",
    with_loopback: bool = False,
    fanout_length: float = 0.0,
    grating_coupler: ComponentSpec = gc_te1550,
    cross_section: CrossSectionSpec = "strip",
    straight: ComponentSpec = "straight",
    taper: ComponentSpec | None = None,
    **kwargs,
) -> Component:
    """Returns component with grating couplers and labels on each port.

    Routes all component ports south.
    Can add align_ports loopback reference structure on the edges.

    Args:
        component: to connect.
        component_name: for the label.
        gc_port_name: grating coupler input port name 'o1'.
        with_loopback: True, adds loopback structures.
        fanout_length: None  # if None, automatic calculation of fanout length.
        grating_coupler: grating coupler instance, function or list of functions.
        cross_section: spec.
        straight: straight component.
        taper: taper component.
        kwargs: cross_section settings.

    """
    c = gf.Component()

    ref = c << gf.routing.add_fiber_array(
        straight=straight,
        bend=bend,
        component=component,
        component_name=component_name,
        grating_coupler=grating_coupler,
        gc_port_name=gc_port_name,
        with_loopback=with_loopback,
        fanout_length=fanout_length,
        cross_section=cross_section,
        taper=taper,
        **kwargs,
    )
    ref.drotate(-90)
    c.add_ports(ref.ports)
    c.copy_child_info(component)

    component_name = component_name or component.name
    grating_coupler = gf.get_component(grating_coupler)
    label = get_input_label_text(gc=grating_coupler, component_name=component_name)
    c.add_label(position=c.ports["o1"].dcenter, text=label, layer=LAYER.TEXT)
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
    s = gf.components.straight(length=l1, cross_section="strip")
    g = c << gf.components.dbr(
        w1=w0 - dw / 2,
        w2=w0 + dw / 2,
        n=n,
        l1=l1,
        l2=l2,
        cross_section="strip",
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
def terminator_short(width2=0.1) -> gf.Component:
    c = gf.Component()
    s = gf.components.taper(cross_section="strip", width2=width2)
    s1 = c << s
    c.add_port("o1", port=s1.ports["o1"])
    c = add_pins_bbox_siepic(c)
    c.flatten()
    return c


@gf.cell
def dbr(
    w0: float = 0.5,
    dw: float = 0.1,
    n: int = 100,
    l1: float = L,
    l2: float = L,
    cross_section: CrossSectionSpec = "strip",
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


@gf.cell
def coupler(**kwargs) -> gf.Component:
    c = gf.components.coupler(**kwargs).dup()
    c.flatten()
    return c


@gf.cell
def mmi1x2(**kwargs) -> gf.Component:
    return gf.components.mmi1x2(**kwargs)


@gf.cell
def dbr_cavity(dbr=dbr, coupler="coupler", **kwargs) -> gf.Component:
    dbr = dbr(**kwargs)
    return gf.components.cavity(component=dbr, coupler=coupler)


@gf.cell
def dbr_cavity_te(component="dbr_cavity", **kwargs) -> gf.Component:
    component = gf.get_component(component, **kwargs)
    return add_fiber_array(component=component)


@gf.cell
def spiral(
    length: float = 100,
    spacing: float = 3.0,
    n_loops: int = 6,
) -> gf.Component:
    """Returns spiral component.

    Args:
        length: length.
        spacing: spacing.
        n_loops: number of loops.
    """
    return gf.c.spiral(
        length=length,
        spacing=spacing,
        n_loops=n_loops,
        bend=bend_euler,
        straight=straight,
    )


coupler90 = partial(gf.components.coupler90, bend=bend_euler, straight=straight)
coupler_straight = partial(
    gf.components.coupler_straight, gap=0.2, cross_section="strip"
)


@gf.cell
def coupler_ring(
    gap: float = 0.2,
    radius: float = 10.0,
    length_x: float = 4.0,
    length_extension: float = 3,
    bend=bend,
    cross_section="strip",
    **kwargs,
) -> Component:
    c = gf.components.coupler_ring(
        gap=gap,
        radius=radius,
        length_x=length_x,
        length_extension=length_extension,
        bend=bend,
        cross_section=cross_section,
        **kwargs,
    ).dup()
    c = add_pins_bbox_siepic(c)
    c.flatten()
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
    res_ref.connect("e1", IDCwithstubs_ref.ports["o2"], allow_type_mismatch=True)

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
        radius: radius of the bends on the nanowire. optical radius will be calculated.
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
    mw_radius = radius
    op_radius = radius - gap - 0.5 * mw_xs.width - 0.5 * op_xs.width

    mw_bend_path = partial(bend, radius=mw_radius)

    op_bend_path = partial(bend, radius=op_radius)
    op_bend = partial(op_bend_path().extrude, cross_section=op_cross_section)

    length_remainder = length_mw - 4 * mw_bend_path().length() - 2 * length_y - length_x

    c = gf.Component()
    op_res = c << ring_single_mod_coupler(
        gap=op_gap,
        radius=op_radius,
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
        + op_radius
        + mw_radius
        + op_xs.width / 2
        + mw_xs.width / 2
        + gap
    )

    c.add_port("e1", port=mw_res.ports["e1"])
    c.add_port("o1", port=op_res.ports["o1"])
    c.add_port("o2", port=op_res.ports["o2"])
    return c


@gf.cell
def ring_single(
    gap: float = 0.2,
    radius: float = 10.0,
    length_x: float = 4.0,
    length_y: float = 0.6,
) -> Component:
    return gf.components.ring_single(
        gap=gap,
        radius=radius,
        length_x=length_x,
        length_y=length_y,
        cross_section="strip",
        bend=bend,
        coupler_ring=coupler_ring,
    )


@gf.cell
def ring_double(
    gap: float = 0.2,
    radius: float = 10.0,
    length_x: float = 4.0,
    length_y: float = 0.6,
) -> Component:
    return gf.components.ring_double(
        gap=gap,
        radius=radius,
        length_x=length_x,
        length_y=length_y,
        cross_section="strip",
        bend=bend,
        coupler_ring=coupler_ring,
    )


ring_double_heater = partial(
    gf.components.ring_double_heater,
    via_stack="via_stack_heater_mtop",
    straight=straight,
    length_y=0.2,
    cross_section_heater="heater_metal",
    cross_section_waveguide_heater="strip_heater_metal",
    cross_section="strip",
    coupler_ring=coupler_ring,
)
ring_single_heater = partial(
    gf.components.ring_single_heater,
    via_stack="via_stack_heater_mtop",
    straight=straight,
    cross_section_heater="heater_metal",
    cross_section_waveguide_heater="strip_heater_metal",
    cross_section="strip",
    coupler_ring=coupler_ring,
)


ebeam_dc_te1550 = partial(
    gf.components.coupler,
)
taper = partial(gf.components.taper)
ring_with_crossing = partial(
    gf.components.ring_single_dut,
    component=ebeam_crossing4_2ports,
    port_name="o4",
    bend=bend,
    cross_section="strip",
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
    buffer: float = 100,
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
        port_names = [port.name for port in component.ports]
        raise ValueError(f"No port {port_name!r} in {port_names}")

    component.add_label(
        text=text, position=component.ports[port_name].dcenter, layer=LAYER.TEXT
    )
    return component


pad_array = partial(gf.components.pad_array, pad=pad, column_pitch=125, row_pitch=125)
add_pads_rf = partial(
    gf.routing.add_electrical_pads_top,
    component="ring_single_heater",
    pad_array="pad_array",
)
add_pads_top = partial(
    gf.routing.add_pads_top,
    component=straight_heater_metal,
)
add_pads_bot = partial(
    gf.routing.add_pads_bot,
    component=straight_heater_metal,
)


@cache
def add_fiber_array_pads_rf(
    component: ComponentSpec = "ring_single_heater",
    username: str = CONFIG.username,
    orientation: float = 0,
    pad_yspacing: float = 50,
    component_name: str | None = None,
    **kwargs,
) -> Component:
    """Returns fiber array with label and electrical pads.

    Args:
        component: to add fiber array and pads.
        username: for the label.
        orientation: for adding pads.
        pad_yspacing: for adding pads.
        component_name: for the label.
        kwargs: for add_fiber_array.
    """

    c0 = gf.get_component(component)
    component_name = component_name or c0.name
    component_name = clean_name(component_name)
    text = f"elec_{username}-{component_name}_G"
    c1 = add_pads_rf(component=c0, orientation=orientation, spacing=(0, pad_yspacing))

    add_label_electrical(component=c1, text=text)
    # ports_names = [port.name for port in c0.ports if port.orientation == orientation]
    # c1 = add_pads_top(component=c0, port_names=ports_names)
    return add_fiber_array(component=c1, component_name=component_name, **kwargs)


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
    text = f"elec_{username}-{clean_name(c0.name)}_G"
    c0 = add_label_electrical(c0, text=text)
    return add_pads_rf(component=c0, **kwargs)


if __name__ == "__main__":
    # c = straight_heater_metal()
    # c = thermal_phase_shifter0()
    # c = straight_one_pin()
    # c = ebeam_adiabatic_te1550()
    # c = ebeam_bdc_te1550()
    # c = gc_tm1550()
    # c = spiral()
    # c = add_pads_top()

    # c.pprint_ports()
    # c.pprint_ports()
    # c = straight()
    # c = terminator_short()
    # c = add_fiber_array_pads_rf(c)

    # c = ring_double(length_y=10)
    # c = ring_with_crossing()
    # c = straight_heater_metal()
    # c = add_fiber_array(straight_heater_metal)
    # c.pprint_ports()
    # c = coupler_ring()
    # c = dbr_cavity_te()
    # c = dbr_cavity()
    # c = ring_single(radius=12)
    # c = bend_euler()
    # c = mzi()
    # c = spiral()
    # c = pad_array()
    # c = bend_euler()
    # c = mzi_heater()
    # c = ring_with_crossing()
    # c = ring_single()
    # c = ring_double()
    # c = ring_double(radius=12, length_x=2, length_y=2)
    # c = straight()
    c = add_fiber_array_pads_rf(component_name="ring_single_heater")
    c.show()
