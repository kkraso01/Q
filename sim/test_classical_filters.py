import numpy as np
from sim.classical_filters import CuckooFilter, XORFilter, VacuumFilter

def test_cuckoo_filter():
    f = CuckooFilter(m=32)
    items = [b"a", b"b", b"c", b"d"]
    for x in items:
        assert f.insert(x)
    for x in items:
        assert f.contains(x)
    assert not f.contains(b"z")

def test_xor_filter():
    f = XORFilter(m=32, k=3)
    items = [b"a", b"b", b"c", b"d"]
    for x in items:
        f.insert(x)
    for x in items:
        assert f.contains(x)
    assert not f.contains(b"z")

def test_vacuum_filter():
    f = VacuumFilter(m=32, k=3)
    items = [b"a", b"b", b"c", b"d"]
    for x in items:
        f.insert(x)
    for x in items:
        assert f.contains(x)
    f.delete(b"a")
    assert not f.contains(b"a")
    assert not f.contains(b"z")
