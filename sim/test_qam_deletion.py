import numpy as np
from sim.qam import QAM

def test_qam_deletion_fp():
    m, k = 8, 2
    qam = QAM(m, k)
    items = [b"a", b"b", b"c"]
    qam.inserted_items = items.copy()
    deleted = [b"b"]
    # Query deleted item: should have low expectation
    exp_deleted = qam.query(items, b"b", shots=256, deleted_items=deleted)
    # Query present item: should have high expectation
    exp_present = qam.query(items, b"a", shots=256, deleted_items=deleted)
    # Query absent item: should have low expectation
    exp_absent = qam.query(items, b"z", shots=256, deleted_items=deleted)
    assert exp_present > 0.4
    assert exp_deleted < 0.2
    assert exp_absent < 0.2
