import numpy as np
from sim.qam import QAM

def run_deletion_sweep():
    results = []
    for theta in [np.pi/4, np.pi/2, np.pi/3]:
        for m in [4, 8, 16]:
            for k in [2, 3]:
                qam = QAM(m, k, theta=theta)
                items = [bytes([i]) for i in range(1, k+2)]
                deleted = [items[0]]
                exp_deleted = qam.query(items, items[0], shots=512, deleted_items=deleted)
                exp_present = qam.query(items, items[1], shots=512, deleted_items=deleted)
                exp_absent = qam.query(items, b"z", shots=512, deleted_items=deleted)
                results.append({
                    'theta': theta,
                    'm': m,
                    'k': k,
                    'exp_deleted': exp_deleted,
                    'exp_present': exp_present,
                    'exp_absent': exp_absent
                })
    return results

def test_deletion_sweep():
    results = run_deletion_sweep()
    for r in results:
        assert r['exp_present'] > 0.3, f"Present item too low: {r}"
        assert r['exp_absent'] < 0.3, f"Absent item too high: {r}"
        # For deletion, expectation should be closer to absent than present
        assert abs(r['exp_deleted'] - r['exp_absent']) < abs(r['exp_deleted'] - r['exp_present']), f"Deleted not closer to absent: {r}"
