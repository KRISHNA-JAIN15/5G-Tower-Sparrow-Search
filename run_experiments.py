import sys
import types
import time
import numpy as np
import math

# Fix for code that expects `np.math` (Paper.py uses `np.math.gamma`)
np.math = math
import pandas as pd
import json

# Create a lightweight dummy streamlit module to avoid UI side-effects when importing Paper.py
class _DummyExpander:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False
    def __call__(self, *a, **k):
        return self
    def __getattr__(self, name):
        return lambda *a, **k: None

class _DummyColumn:
    def __getattr__(self, name):
        return lambda *a, **k: None
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False


class _DummySidebar:
    def columns(self, n):
        # return n dummy column objects
        return tuple(_DummyColumn() for _ in range(n))

    def number_input(self, *a, **k):
        # signature: (label, min, max, default)
        if len(a) >= 4:
            return a[3]
        if len(a) >= 3:
            return a[2]
        return k.get('value', 0)

    def slider(self, *a, **k):
        # return middle/default value
        if 'value' in k:
            return k['value']
        if len(a) >= 4:
            return a[3]
        if len(a) >= 3:
            return (a[1] + a[2]) / 2 if isinstance(a[1], (int, float)) and isinstance(a[2], (int, float)) else a[2]
        return 0

    def checkbox(self, *a, **k):
        return k.get('value', False)

    def number_input_float(self, *a, **k):
        return self.number_input(*a, **k)

    def __getattr__(self, name):
        return lambda *a, **k: None


class _DummyStreamlit:
    def __getattr__(self, name):
        if name == 'expander':
            return lambda *a, **k: _DummyExpander()
        if name == 'sidebar':
            return _DummySidebar()
        return lambda *a, **k: None

# Place dummy into sys.modules so `import streamlit as st` in Paper.py won't error or run UI
sys.modules['streamlit'] = _DummyStreamlit()

# Import the EnhancedSparrowSearchAlgorithm from Paper.py
try:
    from Paper import EnhancedSparrowSearchAlgorithm
except Exception as e:
    print("Failed to import EnhancedSparrowSearchAlgorithm from Paper.py:", e)
    raise


# Helper: random initializer for positions
def random_initial_positions(num_sparrows, num_stations, area_size):
    L1, L2 = area_size
    positions = []
    for _ in range(num_sparrows):
        stations = np.column_stack((np.random.uniform(0, L1, size=num_stations),
                                    np.random.uniform(0, L2, size=num_stations)))
        positions.append(stations)
    return np.array(positions)


# Variant subclasses
class SSA_Base(EnhancedSparrowSearchAlgorithm):
    # Plain baseline: random init, no potential forces, no OBL
    def initialize_positions(self):
        return random_initial_positions(self.num_sparrows, self.num_stations, self.area_size)

    def calculate_potential_forces(self, *a, **k):
        return np.array([0.0, 0.0])

    def opposition_based_learning_jump(self, *a, **k):
        return None


class SSA_Chaotic(EnhancedSparrowSearchAlgorithm):
    # Chaotic init only: keep Enhanced initializer, disable potential & OBL
    def calculate_potential_forces(self, *a, **k):
        return np.array([0.0, 0.0])

    def opposition_based_learning_jump(self, *a, **k):
        return None


class SSA_Potential(EnhancedSparrowSearchAlgorithm):
    # Potential field navigation only: random init, keep potential, disable OBL
    def initialize_positions(self):
        return random_initial_positions(self.num_sparrows, self.num_stations, self.area_size)

    def opposition_based_learning_jump(self, *a, **k):
        return None


class SSA_OBL(EnhancedSparrowSearchAlgorithm):
    # Opposition-based learning only: random init, disable potential, keep OBL
    def initialize_positions(self):
        return random_initial_positions(self.num_sparrows, self.num_stations, self.area_size)

    def calculate_potential_forces(self, *a, **k):
        return np.array([0.0, 0.0])


# The full-featured class (all improvements) will be used as-is: EnhancedSparrowSearchAlgorithm


def run_variant(Cls, variant_name, runs=3, **ssa_kwargs):
    results = []
    for r in range(runs):
        ssa = Cls(**ssa_kwargs)
        t0 = time.time()
        best_pos, best_fit = ssa.optimize()
        dt = time.time() - t0
        results.append({
            'variant': variant_name,
            'run': r + 1,
            'fitness': float(best_fit),
            'time_s': dt
        })
        print(f"{variant_name} run {r+1}: fitness={best_fit:.4f}, time={dt:.2f}s")
    return results


def main():
    # Experiment settings (kept moderate for speed)
    ssa_kwargs = dict(
        num_sparrows=30,
        num_stations=4,
        area_size=(10, 10),
        grid_size=(50, 50),
        Rs=2.0,
        max_iter=100,
        seed=42,
        early_stop_iter=25,
        boundary_buffer=1.0,
        num_pits=3,
        pit_sizes=[1.2, 1.2, 1.2]
    )

    variants = [
        (SSA_Base, 'Baseline (base)'),
        (SSA_Chaotic, 'Base + Chaotic Init'),
        (SSA_Potential, 'Base + Potential Field'),
        (SSA_OBL, 'Base + Opposition-Based Learning'),
        (EnhancedSparrowSearchAlgorithm, 'All Combined')
    ]

    all_results = []
    for Cls, name in variants:
        print('\nRunning variant:', name)
        res = run_variant(Cls, name, runs=3, **ssa_kwargs)
        all_results.extend(res)

    # Save results to CSV and JSON
    df = pd.DataFrame(all_results)
    out_csv = 'optimization_results/variants_results.csv'
    df.to_csv(out_csv, index=False)
    with open('optimization_results/variants_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary table (mean and std per variant)
    summary = df.groupby('variant').fitness.agg(['mean', 'std', 'count']).reset_index()
    print('\nSummary:')
    print(summary.to_string(index=False))

    print(f"\nDetailed results saved to {out_csv} and optimization_results/variants_results.json")


if __name__ == '__main__':
    main()
