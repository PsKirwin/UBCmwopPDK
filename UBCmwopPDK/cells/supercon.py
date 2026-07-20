"""Superconducting circuits. author: pkirwin@ece.ubc.ca"""
from functools import partial

import gdsfactory as gf
from gdsfactory.path import transition
from gdsfactory.typings import AngleInDegrees, ComponentSpec, CrossSectionSpec, Float2

from UBCmwopPDK import cells
from UBCmwopPDK.tech import TECH, LAYER
import numpy as np

@gf.cell
def pad_supercon(
    size: tuple[float, float] = (400.0, 400.0),
    gap: float = 130,
    buffer: float = 100,
    cross_section: CrossSectionSpec = "supercon_CPW_feedline",
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
    xsec_pad = gf.CrossSection(sections=(s0, s1, s2))

    xsec_feedline = gf.get_cross_section(cross_section)

    pad = gf.path.extrude(p=gf.path.straight(length=size[0]), cross_section=xsec_pad)
    taper = gf.components.taper_cross_section_linear(
        cross_section1=xsec_pad, cross_section2=xsec_feedline, length=size[0]
    )
    c = gf.Component()
    pad_ref = c << pad
    taper_ref = c << taper
    taper_ref.connect("e1", pad_ref.ports["e2"])
    c.add_port("e1", port=taper_ref.ports["e2"],port_type="electrical")
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

@gf.cell
def supercon_CPW_resonator_IDC(
    coupler_spec: dict | None = None,
    length: float = 5000,  # length in [um]
    cross_section: CrossSectionSpec = "supercon_CPW",
    cap_cross_section: CrossSectionSpec = "supercon_CPW_cap",
    label: str | None = None,
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
    fingers = IDC.settings.fingers
    finger_length = IDC.settings.finger_length
    finger_gap = IDC.settings.finger_gap
    thickness = IDC.settings.thickness
    stub1_length = float(np.max([3 * thickness, 30]))
    stub2_length = float(np.max([3 * thickness, 10]))
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

    IDCwithstubs = gf.Component("IDCwithstubs")
    IDC_ref = IDCwithstubs << IDC
    stub1_ref = IDCwithstubs << stub1
    stub2_ref = IDCwithstubs << stub2
    IDC_ref.connect("o1", stub1.ports["o2"], allow_width_mismatch=True)
    stub2_ref.connect("o1", IDC_ref.ports["o2"], allow_width_mismatch=True)

    xsize = finger_length + finger_gap + 2 * thickness + stub1_length + stub2_length
    ysize = fingers * thickness + (fingers - 1) * finger_gap + 20
    keepout_box = gf.components.rectangle(size=(xsize, ysize), layer=trace_layer)

    temp = gf.Component("temp")
    keepout_box_ref = temp << keepout_box
    keepout_box_ref.movey(-ysize / 2)
    keepout_gap = gf.boolean(
        B=IDCwithstubs,
        A=keepout_box_ref,
        operation="not",
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
        label_inst = gf.components.text(text=cellname, size=30, layer=gap_layer)
    else:
        cellname = "Res_" + str(int(length)) + "um"
        label_inst = gf.components.text(text="", size=30, layer=gap_layer)
    # putting in parent cell
    c = gf.Component(cellname)
    IDCwithstubs_ref = c << IDCwithstubs
    res_ref = c << res
    cap_ref = c << cap
    label_ref = c << label_inst
    label_ref.rotate(90).movex(6 * (finger_length + stub1_length)).movey(2 * ysize)
    res_ref.connect("e1", IDCwithstubs_ref.ports["o2"], allow_type_mismatch=True)
    cap_ref.connect("e1", res_ref.ports["e2"], allow_layer_mismatch=True)

    c.add_port("e1", port=IDCwithstubs_ref.ports["o1"])

    return c


@gf.cell
def supercon_wire_resonator_IDC(
    coupler_spec: dict | None = None,
    lengths: list[float] = [700, 1000, 1000, 500],  # length in [um]
    bend_path = partial(gf.path.euler, use_eff=True),
    bend_radius: float | None = 100,  # [um]
    wire_path: gf.Path | None = None,
    cross_section: CrossSectionSpec = "supercon_wire",
    label: str | None = None,
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
    fingers = IDC.settings.fingers
    finger_length = IDC.settings.finger_length
    finger_gap = IDC.settings.finger_gap
    thickness = IDC.settings.thickness
    stub1_length = float(np.max([3 * thickness, 30]))
    stub2_length = float(np.max([3 * thickness, 10]))
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

    IDCwithstubs = gf.Component()
    IDC_ref = IDCwithstubs << IDC
    stub1_ref = IDCwithstubs << stub1
    stub2_ref = IDCwithstubs << stub2
    IDC_ref.connect("o1", stub1.ports["o2"], allow_width_mismatch=True)
    stub2_ref.connect("o1", IDC_ref.ports["o2"],allow_width_mismatch=True)

    xsize = finger_length + finger_gap + 2 * thickness + stub1_length + stub2_length
    ysize = fingers * thickness + (fingers - 1) * finger_gap + 20
    keepout_box = gf.components.rectangle(size=(xsize, ysize), layer=trace_layer)

    temp = gf.Component()
    keepout_box_ref = temp << keepout_box
    keepout_box_ref.movey(-ysize / 2)
    keepout_gap = gf.boolean(
        B=IDCwithstubs,
        A=keepout_box_ref,
        operation="not",
        layer=gap_layer,
    )
    _keepout_gap_ref = IDCwithstubs << keepout_gap

    IDCwithstubs.add_port("o1", port=stub1_ref.ports["o1"])
    IDCwithstubs.add_port("o2", port=stub2_ref.ports["o2"])

    # resonator path and extrude

    bend_path = (
            partial(bend_path, radius=cross_section.radius)
        if pass_cross_section_to_bend
        else partial(bend_path,radius=bend_radius)
    )
    north_straight = gf.path.straight(length=lengths[0])
    east_straight = gf.path.straight(length=lengths[1])
    south_straight = gf.path.straight(length=lengths[2])
    west_straight = gf.path.straight(length=lengths[3])

    res_path = gf.Path()
    res_path.append(
        [
            bend_path(angle=90),
            north_straight,
            bend_path(angle=-90),
            east_straight,
            bend_path(angle=-90),
            south_straight,
            bend_path(angle=-90),
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
        label_inst = gf.components.text(text=cellname, size=30, layer=gap_layer)
    else:
        cellname = "Res_" + str(int(length_total)) + "um"
        label_inst = gf.components.text(text="", size=30, layer=gap_layer)
    # putting in parent cell
    c = gf.Component(cellname)
    IDCwithstubs_ref = c << IDCwithstubs
    res_ref = c << res
    label_ref = c << label_inst
    label_ref.rotate(90).movex(6 * finger_length + stub1_length).movey(2 * ysize)
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
    radius: float | None = 10.0,
    length_x: float = 4.0,
    length_y: float = 0.6,
    bend: ComponentSpec = "bend_euler",
    bend_coupler: ComponentSpec | None = None,
    length_coupler: float = 4.0,
    offset_coupler: float = 0.0,
    radius_coupler: float | None = 10,
    straight:  ComponentSpec = "straight",
    cross_section: CrossSectionSpec = "strip",
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
    #cs = straight(length=length_coupler, cross_section=cross_section)
    cs = gf.get_component(straight, length=length_coupler, cross_section=cross_section)
    bc = (
        gf.get_component(bend_coupler, cross_section=cross_section_coupler)
        #bend_coupler(cross_section=cross_section_coupler)
        if pass_cross_section_to_bend
        else gf.get_component(bend_coupler, radius=radius_coupler)
        #bend_coupler(radius=radius_coupler)
    )

    if length_coupler > 0:
        cs_ref = c.add_ref(cs)
    bcl = c << bc
    bcr = c << bc
    bcl.connect("o1", other=cs_ref.ports["o1"])
    bcr.connect("o2", other=cs_ref.ports["o2"])

    # ring
    #sy = straight(length=length_y, cross_section=cross_section)
    sy = gf.get_component(straight, length=length_y, cross_section=cross_section)
    # sx = straight(length=length_x, cross_section=cross_section)
    sx = gf.get_component(straight, length=length_x, cross_section=cross_section)
    b = (
        # bend(cross_section=cross_section)
        gf.get_component(bend, cross_section=cross_section)
        if pass_cross_section_to_bend
        else gf.get_component(bend, radius=radius)
        # bend(radius=radius)
    )
    sl = sy
    sr = sy
    st = sx
    sb = sx

    if length_y > 0:
        sl_ref = c.add_ref(sl)
        sr_ref = c.add_ref(sr)

    if length_x > 0:
        st_ref = c.add_ref(st)
        sb_ref = c.add_ref(sb)

    bul = c << b
    bll = c << b
    bur = c << b
    blr = c << b

    sb_ref.movey(cs.info["width"] + gap)
    sb_ref.movex(-offset_coupler)

    blr.connect("o1", other=sb_ref.ports["o2"])
    sr_ref.connect("o1", other=blr.ports["o2"])
    bur.connect("o1", other=sr_ref.ports["o2"])
    st_ref.connect("o1", other=bur.ports["o2"])
    bul.connect("o1", other=st_ref.ports["o2"])
    sl_ref.connect("o1", other=bul.ports["o2"])
    bll.connect("o1", other=sl_ref.ports["o2"])

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
    bend: gf.Path | partial = partial(gf.path.euler, use_eff=True),
    length_mw: float = 2500,
    op_gap: float = 0.2,
    op_bend_coupler: ComponentSpec | None = None,
    op_length_coupler: float = 4.0,
    op_offset_coupler: float = 0.0,
    op_radius_coupler: float | None = 100,
    op_straight: ComponentSpec = cells.straight,
    op_cross_section: CrossSectionSpec = "strip",
    op_pass_cross_section_to_bend: bool = True,
    mw_coupler_spec: dict | None = None,
    mw_cross_section: CrossSectionSpec = "supercon_wire",
    mw_label: str | None = None,
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
    mw_bend_path = partial(gf.path.euler, radius=mw_radius, use_eff=True)
    #op_bend_path = partial(cells.bend_euler, radius=op_radius)
    op_bend = partial(cells.bend_euler, radius=op_radius, cross_section=op_cross_section)

    length_remainder = length_mw - 4 * mw_bend_path().length() - 2 * length_y - length_x

    c = gf.Component()
    op_res = ring_single_mod_coupler(
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
    op_res_inst = c << op_res

    mw_res = supercon_wire_resonator_IDC(
        coupler_spec=mw_coupler_spec,
        lengths=[length_y, length_x, length_y, length_remainder],
        bend_path=mw_bend_path,
        bend_radius=mw_radius,
        cross_section=mw_cross_section,
        label=mw_label,
        pass_cross_section_to_bend=mw_pass_cross_section_to_bend,
    )
    mw_res_inst = c << mw_res

    op_res_inst.movey(-op_gap - op_xs.width / 2 + mw_xs.width / 2 + gap)
    op_res_inst.movex(
        mw_res.info["coupler_length"]
        + op_radius
        + mw_radius
        + op_xs.width / 2
        + mw_xs.width / 2
        + gap
    )

    c.add_port("e1", port=mw_res_inst.ports["e1"])
    c.add_port("o1", port=op_res_inst.ports["o1"])
    c.add_port("o2", port=op_res_inst.ports["o2"])
    return c

@gf.cell
def hairpin_inductor(
    length: float = 100,
    wire_width: float = 1,
    gap: float = 1,
    cross_section: CrossSectionSpec = "supercon_wire",
) -> gf.Component:
    """
    Hairpin loop of wire.

    Args:
    length: length of the straight sections of the hairpin.
    wire_width: width of the wire.
    gap: gap between the two sides of the hairpin, centre-to-centre.
    cross_section: cross section spec for the hairpin.
    straight: straight spec for the hairpin.
    bend: bend spec for the hairpin.

    todo: add a modification to the cross-section to allow for flow holes on the outer radius
    """
    c = gf.Component()

    p = gf.Path()
    p.append(
        [
            gf.path.straight(length=length),
            gf.path.arc(radius=(gap + wire_width) / 2, angle=180),
            gf.path.straight(length=length),
        ]
    )
    ind = gf.path.extrude(p=p, cross_section=cross_section)

    ind_inst = c << ind

    c.add_port("e1", port=ind_inst.ports["e2"])
    c.add_port("e2", port=ind_inst.ports["e1"])



    return c

@gf.cell
def ART_resonator_capacitor(
    cap_length_x: float = 650,
    cap_length_y: float = 650,
    cap_wire_width: float = 15,
    cap_slab_width: float = 47,
    cap_wire_gap: float = 5,
    cap_bend_radius: float = 100,  # [um]
    cap_cross_section: CrossSectionSpec = "supercon_pair_hole",
    cap_stub_length: float = 50,
) -> gf.Component:
    cap = gf.Component()
    cap_path = gf.Path()
    cap_path.append(
        [
            gf.path.straight(length=cap_stub_length),
            gf.path.arc(radius=cap_bend_radius, angle=-90),
            gf.path.straight(length=cap_length_y),
            gf.path.arc(radius=cap_bend_radius, angle=-90),
            gf.path.straight(length=cap_length_x),
            gf.path.arc(radius=cap_bend_radius, angle=-90),
            gf.path.straight(length=cap_length_y),
            gf.path.arc(radius=cap_bend_radius, angle=-90),
            gf.path.straight(length=cap_stub_length),
        ]
    )

    cap_cross_section_inst = gf.get_cross_section(cap_cross_section, wire_width=cap_wire_width, radius=cap_bend_radius, wire_gap=cap_wire_gap, slab_width=cap_slab_width)
    extrude_path_ref = cap << gf.path.extrude(p=cap_path, cross_section=cap_cross_section_inst)
    cap.add_ports(extrude_path_ref.ports)
    return cap

@gf.cell
def ART_resonator(
    inductor_length: float = 100,
    inductor_wire_width: float = 0.5,
    inductor_slab_width: float = 10,
    inductor_gap: float = 4.5,
    inductor_cross_section: CrossSectionSpec = "supercon_wire_hole",
    cap_length_x: float = 650,
    cap_length_y: float = 650,
    cap_wire_width: float = 15,
    cap_slab_width: float = 47,
    cap_wire_gap: float = 5,
    cap_bend_radius: float = 100,  # [um]
    cap_cross_section: CrossSectionSpec = "supercon_pair_hole",
    label: str | None = None,
    trace_layer=LAYER.SC_TRACE,
    gap_layer=LAYER.SC_GAP,
    keepout_y_adjustment: float = 20,
) -> gf.Component:
    """Returns a superconducting ART resonator, with no coupler.
    Note that a ground plane will need to be defined separately.
    Inspired by Sullivan 2020.

    Args:
        inductor_length: length of the straight sections of the hairpin.
        inductor_wire_width: width of the wire.
        inductor_slab_width: width of the slab on either side of the wire.
        inductor_gap: gap between the two sides of the hairpin, inner edge to inner edge.
        inductor_cross_section: cross section spec for the hairpin.

        cap_length_x: horizontal side length of the capacitor in um
        cap_length_y: vertical side length of the capacitor in um
        cap_wire_width: width of the wires in the capacitor in um
        cap_wire_gap: centre-to-centre gap between the wires in the capacitor in um
        cap_straight: straight spec for the capacitor wires
        cap_bend: bend spec for the capacitor
        cap_bend_radius: bend radius for the capacitor bends.
        cap_cross_section: CrossSectionSpec for the capacitor wire pair.
        label: drawn label to be patterned for identification. If left blank, no label will be drawn
        trace_layer: can change to arbitrary layer
        gap_layer: can change to arbitrary layer
        keepout_y_adjustment: adjustment for the keepout region around the resonator

    author: Phillip Kirwin (pkirwin@ece.ubc.ca)
    """
    c = gf.Component()

    # inductor
    inductor_cross_section_inst = gf.get_cross_section(inductor_cross_section, width=inductor_wire_width, slab_width=inductor_slab_width)
    inductor = hairpin_inductor(
        length=inductor_length,
        wire_width=inductor_wire_width,
        gap=inductor_gap,
        cross_section=inductor_cross_section_inst,
    )

    inductor_inst = c << inductor

    # capacitor parts
    cap_opening_length = 4 * cap_wire_width + inductor_wire_width + inductor_gap
    cap_stub_length = (cap_length_x - cap_opening_length) / 2
    cap_inst = c << ART_resonator_capacitor(
        cap_length_x=cap_length_x,
        cap_length_y=cap_length_y,
        cap_wire_width=cap_wire_width,
        cap_slab_width=cap_slab_width,
        cap_wire_gap=cap_wire_gap,
        cap_bend_radius=cap_bend_radius,
        cap_cross_section=cap_cross_section,
        cap_stub_length=cap_stub_length
    )
    cap_inst.movex(cap_opening_length/2)

    # adding a length of slab to connect the ends of the capacitor
    slab_cross_section = gf.get_cross_section("cross_section", layer=LAYER.SLAB150, width=cap_slab_width)
    slab_straight = gf.get_component("straight", cross_section=slab_cross_section,length=cap_opening_length)
    slab_straight_ref = c << slab_straight
    slab_straight_ref.movex(-cap_opening_length/2)

    # adding a short wire at the ends of the cap leads
    interposer_length = 0.01
    interposer_cross_section_inst = gf.get_cross_section(inductor_cross_section, width=cap_wire_width, slab_width=cap_slab_width/2)
    interposer = gf.path.extrude(p=gf.path.straight(length=interposer_length), cross_section=interposer_cross_section_inst)
    interposer_inst1 = c << interposer
    interposer_inst1.connect("e1", other=cap_inst.ports["e2"])
    interposer_inst2 = c << interposer
    interposer_inst2.connect("e1", other=cap_inst.ports["e3"])

    # # Define a custom polynomial transition function from y1 -> y2, for t ∈ [0,1].
    # def polynomial(t: float, y1: float, y2: float) -> float:
    #     return (y2 - y1) * t**20 + y1
    # hook up inductor to capacitor with a transition
    offset_left = -(cap_wire_width-inductor_wire_width) / 2
    inductor_cross_section_inst = gf.get_cross_section(inductor_cross_section, width=inductor_wire_width, offset=offset_left, slab_width=inductor_slab_width)
    transition1_obj = gf.path.transition(interposer_cross_section_inst, inductor_cross_section_inst)
    transition1_y = 2 * cap_wire_width
    transition1_x = transition1_y - interposer_length + offset_left
    transition1_points = np.array([(0,0),(transition1_x,0),(transition1_x,-transition1_y)])
    transition1_path = gf.path.smooth(
        points=transition1_points,
        radius=transition1_x,
        bend=gf.path.euler,  # Alternatively, use pp.arc, which will create a constant-radius bend.
        use_eff=True,
        # p=0.5,
    )
    # transition1_path.plot()
    transition1 = gf.path.extrude_transition(transition1_path, transition1_obj)
    transition1_inst = c << transition1
    transition1_inst.connect("e1", other=interposer_inst1.ports["e2"])
    inductor_inst.connect("e2", other=transition1_inst.ports["e2"])

    offset_right = -(cap_wire_width-inductor_wire_width) / 2
    inductor_cross_section_inst = gf.get_cross_section(inductor_cross_section, width=inductor_wire_width, offset=offset_right, slab_width=inductor_slab_width)
    transition2_obj = gf.path.transition(inductor_cross_section_inst, interposer_cross_section_inst)
    transition2_y = transition1_y + cap_wire_gap + cap_wire_width
    transition2_x = transition1_x + offset_right - offset_left
    transition2_points = np.array([(0,0),(0,transition2_y),(transition2_x,transition2_y)])
    transition2_path = gf.path.smooth(
        points=transition2_points,
        # radius=transition2_x,
        radius=transition2_x-0.001,
        bend=gf.path.euler,  # Alternatively, use pp.arc, which will create a constant-radius bend.
        use_eff=True,
        # p=1,
    )
    transition2 = gf.path.extrude_transition(transition2_path, transition2_obj)
    transition2_inst = c << transition2
    transition2_inst.connect("e1", other=inductor_inst.ports["e1"])

    # keepout box
    xsize = cap_length_x + 2 * cap_bend_radius + cap_wire_gap + 2*cap_wire_width + 6*cap_wire_gap
    ysize = cap_length_y + 2 * cap_bend_radius + cap_wire_gap + 2*cap_wire_width + 3*cap_wire_gap + keepout_y_adjustment
    print(ysize)
    keepout_box = gf.components.rectangle(size=(xsize, ysize), layer=LAYER.SC_GAP, centered=True)
    keepout_box_ref = c << keepout_box
    keepout_box_ref.movey(-ysize/2 + cap_wire_width + 3.5*cap_wire_gap)

    port_center_y = cap_length_y + cap_wire_gap/2 + cap_wire_width + 2 * cap_bend_radius
    c.add_port('e1',center=(0,-port_center_y),width=10,orientation=270,layer=LAYER.SC_TRACE)

    return c


@gf.cell
def PCC_1D_inline(
    cross_section: CrossSectionSpec = "strip",
    width: float = 0.5,
    cavity_length: float = 0.56,
    front_wg_length: float = 20.0,
    back_wg_length: float = 10.0,
    n_max_backmirror: int = 9,
    n_taper_backmirror: int = 4,
    radius_max_backmirror: float = 0.1,
    radius_min_backmirror: float = 0.05,
    n_max_frontmirror: int = 1,
    n_taper_frontmirror: int = 4,
    radius_max_frontmirror: float = 0.07,
    radius_min_frontmirror: float = 0.05,
    pitch_scale: float = 1.22,
    pitch_offset: float = 0.308,
    hole_layer=LAYER.WG_KEEPOUT,

) -> gf.Component:
    """
    Returns an 1D photonic crystal cavity meant to be measured in reflection via the front
    mirror. The front mirror has tapers on both sides, while the back mirror only has tapers on the cavity side. The pitch of the holes is determined by a linear function of the hole radius, with parameters given by pitch_scale and pitch_offset. The cavity is centered at x=0.

    Args:
        cross_section: CrossSectionSpec for the waveguide
        width: width of the waveguide in um
        cavity_length: length of the cavity region in um
        front_wg_length: length of the front waveguide in um
        back_wg_length: length of the back waveguide in um
        n_max_backmirror: number of full-sized holes (back mirror)
        n_taper_backmirror: number of tapered holes (back mirror)
        radius_max_backmirror: radius of the full-sized holes (back mirror)
        radius_min_backmirror: radius of the smallest tapered hole (back mirror)
        n_max_frontmirror: number of full-sized holes (front mirror)
        n_taper_frontmirror: number of tapered holes (front mirror)
        radius_max_frontmirror: radius of the full-sized holes (front mirror)
        radius_min_frontmirror: radius of the smallest tapered hole (front mirror)
        pitch_scale: linear pitch scale (the m in y=mx+b) (unitless)
        pitch_offset: linear pitch offset (the b in y=mx+b) in um
        hole_layer: layer for the holes


    author: Phillip Kirwin (pkirwin@ece.ubc.ca)
    """
    c = gf.Component()

    # back mirror
    backmirror_c = gf.Component(name="back_mirror")
    cumulative_pitch = -cavity_length / 2
    radius = radius_min_backmirror
    for i in range(n_max_backmirror + n_taper_backmirror):
        if i == 0:
            pitch = 0
        else:
            pitch = pitch_scale * radius + pitch_offset
        cumulative_pitch -= pitch
        if i < n_taper_backmirror:
            radius = radius_min_backmirror + (radius_max_backmirror - radius_min_backmirror) * i / n_taper_backmirror
        else:
            radius = radius_max_backmirror
        hole = gf.components.circle(radius=radius, layer=hole_layer, angle_resolution=0.1)
        hole_ref = backmirror_c << hole
        hole_ref.movex(cumulative_pitch)

    backmirror_inst = c << backmirror_c

    # front mirror
    frontmirror_c = gf.Component(name="front_mirror")
    cumulative_pitch = cavity_length / 2
    radius = radius_min_frontmirror
    for i in range(n_max_frontmirror + 2 * n_taper_frontmirror):
        if i == 0:
            pitch = 0
        else:
            pitch = pitch_scale * radius + pitch_offset
        cumulative_pitch += pitch
        if i < n_taper_frontmirror:
            radius = radius_min_frontmirror + (radius_max_frontmirror - radius_min_frontmirror) * i / n_taper_frontmirror
        elif i >= n_taper_frontmirror and i < n_taper_frontmirror + n_max_frontmirror:
            radius = radius_max_frontmirror
        elif i >= n_taper_frontmirror + n_max_frontmirror:
            radius = radius_max_frontmirror - (radius_max_frontmirror - radius_min_frontmirror) * (i + 1 - (n_taper_frontmirror + n_max_frontmirror)) / n_taper_frontmirror

        hole = gf.components.circle(radius=radius, layer=hole_layer, angle_resolution=0.1)
        hole_ref = frontmirror_c << hole
        hole_ref.movex(cumulative_pitch)

    frontmirror_inst = c << frontmirror_c

    back_waveguide = gf.get_component("straight", cross_section=cross_section, length=back_wg_length, width=width)
    back_waveguide_inst = c << back_waveguide

    front_waveguide = gf.get_component("straight", cross_section=cross_section, length=front_wg_length, width=width)
    front_waveguide_inst = c << front_waveguide

    back_waveguide_inst.connect("o2", front_waveguide_inst.ports["o1"])

    c.add_port("o1", port=front_waveguide_inst.ports["o2"])
    c.add_port("o2", port=back_waveguide_inst.ports["o1"])

    return c

@gf.cell
def magnolia_transducer(
    pcc_cross_section: CrossSectionSpec = "rib",
    pcc_width: float = 0.5,
    pcc_cavity_length: float = 100,
    pcc_front_length: float = 80,
    pcc_back_length: float = 56,
    pcc_n_max_backmirror: int = 9,
    pcc_n_taper_backmirror: int = 4,
    pcc_radius_max_backmirror: float = 0.1,
    pcc_radius_min_backmirror: float = 0.05,
    pcc_n_max_frontmirror: int = 1,
    pcc_n_taper_frontmirror: int = 4,
    pcc_radius_max_frontmirror: float = 0.07,
    pcc_radius_min_frontmirror: float = 0.05,
    pcc_pitch_scale: float = 1.22,
    pcc_pitch_offset: float = 0.308,
    pcc_hole_layer=LAYER.WG_KEEPOUT,
    inductor_length: float = 100,
    inductor_wire_width: float = 0.5,
    inductor_slab_width: float = 10,
    inductor_gap: float = 4.5,
    inductor_cross_section: CrossSectionSpec = "supercon_wire_hole",
    cap_length_x: float = 650,
    cap_length_y: float = 650,
    cap_wire_width: float = 15,
    cap_slab_width: float = 47,
    cap_wire_gap: float = 5,
    cap_bend_radius: float = 100,  # [um]
    cap_cross_section: CrossSectionSpec = "supercon_pair_hole",
    label: str | None = None,
    trace_layer=LAYER.SC_TRACE,
    gap_layer=LAYER.SC_GAP,
    keepout_y_adjustment: float = 20,

) -> gf.Component:
    c = gf.Component()

    SC_resonator = ART_resonator(
        inductor_length=inductor_length,
        inductor_wire_width=inductor_wire_width,
        inductor_slab_width=inductor_slab_width,
        inductor_gap=inductor_gap,
        inductor_cross_section=inductor_cross_section,
        cap_length_x=cap_length_x,
        cap_length_y=cap_length_y,
        cap_wire_width=cap_wire_width,
        cap_wire_gap=cap_wire_gap,
        cap_slab_width=cap_slab_width,
        cap_bend_radius=cap_bend_radius,
        cap_cross_section=cap_cross_section,
        label=label,
        trace_layer=trace_layer,
        gap_layer=gap_layer,
        keepout_y_adjustment=keepout_y_adjustment,
    )
    SC_resonator_inst = c << SC_resonator

    PCC = PCC_1D_inline(
        cross_section=pcc_cross_section,
        width=pcc_width,
        cavity_length=pcc_cavity_length,
        front_wg_length=pcc_front_length,
        back_wg_length=pcc_back_length,
        n_max_backmirror=pcc_n_max_backmirror,
        n_taper_backmirror=pcc_n_taper_backmirror,
        radius_max_backmirror=pcc_radius_max_backmirror,
        radius_min_backmirror=pcc_radius_min_backmirror,
        n_max_frontmirror=pcc_n_max_frontmirror,
        n_taper_frontmirror=pcc_n_taper_frontmirror,
        radius_max_frontmirror=pcc_radius_max_frontmirror,
        radius_min_frontmirror=pcc_radius_min_frontmirror,
        pitch_scale=pcc_pitch_scale,
        pitch_offset=pcc_pitch_offset,
        hole_layer=pcc_hole_layer,
    )
    PCC_inst = c << PCC
    PCC_inst.rotate(90)
    PCC_inst.movey(-200+127)

    c.add_port("o1", port=PCC_inst.ports["o1"])
    c.add_port("e1", port=SC_resonator_inst.ports["e1"])
    return c
