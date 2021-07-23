from ubc.import_gds import import_gds
from pp.component import Component
from pp.components.ring_single_dut import ring_single_dut


def crossing() -> Component:
    """TE waveguide crossing."""
    return import_gds("ebeam_crossing4", rename_ports=True)


def ring_with_crossing(**kwargs) -> Component:
    return ring_single_dut(component=crossing(), **kwargs)


if __name__ == "__main__":
    c = ring_with_crossing()
    c.show()