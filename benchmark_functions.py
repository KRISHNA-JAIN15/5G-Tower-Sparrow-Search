"""
Benchmark SSA variants on standard optimization test functions.
"""
import sys
import time
import math
import numpy as np
import pandas as pd
import json

# Fix for np.math usage in Paper.py
np.math = math

# Dummy streamlit stubs
class _DummyExpander:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __call__(self, *a, **k): return self
    def __getattr__(self, name): return lambda *a, **k: None

class _DummyColumn:
    def __getattr__(self, name): return lambda *a, **k: None
    def __enter__(self): return self
    def __exit__(self, *a): return False

class _DummySidebar:
    def columns(self, n): return tuple(_DummyColumn() for _ in range(n))
    def number_input(self, *a, **k):
        if len(a) >= 4: return a[3]
        if len(a) >= 3: return a[2]
        return k.get('value', 0)
    def slider(self, *a, **k):
        if 'value' in k: return k['value']
        if len(a) >= 4: return a[3]
        return 0
    def checkbox(self, *a, **k): return k.get('value', False)
    def __getattr__(self, name): return lambda *a, **k: None

class _DummyStreamlit:
    def __getattr__(self, name):
        if name == 'expander': return lambda *a, **k: _DummyExpander()
        if name == 'sidebar': return _DummySidebar()
        return lambda *a, **k: None

sys.modules['streamlit'] = _DummyStreamlit()

# ============== Benchmark Functions ==============

def sphere_function(x):
    return np.sum(x ** 2)

def rosenbrock_function(x):
    return sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

def ackley_function(x):
    a, b, c = 20, 0.2, 2 * np.pi
    d = len(x)
    return -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / d)) - np.exp(np.sum(np.cos(c * x)) / d) + a + np.exp(1)

def rastrigin_function(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def griewank_function(x):
    sum_sq = np.sum(x**2) / 4000
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_sq - prod_cos + 1

def schwefel_function(x):
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

# ============== Generic SSA for Benchmark Functions ==============

class BenchmarkSSA:
    """
    A simplified SSA for optimizing benchmark functions.
    Minimization problem.
    """
    def __init__(self, fitness_fn, dim=30, bounds=(-100, 100),
                 num_sparrows=30, max_iter=500, seed=None,
                 use_chaotic_init=False, use_potential_field=False, use_obl=False):
        self.fitness_fn = fitness_fn
        self.dim = dim
        self.lb, self.ub = bounds
        self.num_sparrows = num_sparrows
        self.max_iter = max_iter
        self.use_chaotic_init = use_chaotic_init
        self.use_potential_field = use_potential_field
        self.use_obl = use_obl

        if seed is not None:
            np.random.seed(seed)

        # Initialize positions
        self.positions = self._initialize_positions()
        self.fitness = np.array([self.fitness_fn(p) for p in self.positions])

        self.best_idx = np.argmin(self.fitness)
        self.best_pos = self.positions[self.best_idx].copy()
        self.best_fit = self.fitness[self.best_idx]

        self.convergence = []

    def _initialize_positions(self):
        if self.use_chaotic_init:
            # Chaotic initialization using logistic map
            positions = np.zeros((self.num_sparrows, self.dim))
            x = np.random.rand()
            for i in range(self.num_sparrows):
                for j in range(self.dim):
                    x = 4.0 * x * (1 - x)
                    positions[i, j] = self.lb + x * (self.ub - self.lb)
            return positions
        else:
            return np.random.uniform(self.lb, self.ub, (self.num_sparrows, self.dim))

    def _potential_force(self, pos, t):
        """Potential field: attract toward current best with decaying strength."""
        if not self.use_potential_field:
            return np.zeros(self.dim)
        # Decaying attraction strength
        strength = 0.5 * (1 - t / self.max_iter)
        attract = strength * (self.best_pos - pos)
        # Clip to avoid extreme values
        return np.clip(attract, -1.0, 1.0)

    def _opposition_based_learning(self):
        """OBL: check opposite position for a subset of sparrows."""
        if not self.use_obl:
            return
        candidates = np.random.choice(self.num_sparrows, max(1, int(0.2 * self.num_sparrows)), replace=False)
        for i in candidates:
            opposite = self.lb + self.ub - self.positions[i]
            opposite = np.clip(opposite, self.lb, self.ub)
            opp_fit = self.fitness_fn(opposite)
            if opp_fit < self.fitness[i]:
                self.positions[i] = opposite
                self.fitness[i] = opp_fit

    def optimize(self):
        ST = 0.8  # Safety threshold
        PD = 0.2  # Proportion of producers

        for t in range(self.max_iter):
            sorted_idx = np.argsort(self.fitness)
            num_producers = max(1, int(PD * self.num_sparrows))
            producers = sorted_idx[:num_producers]
            scroungers = sorted_idx[num_producers:]

            R2 = np.random.rand()
            # Update producers
            for i in producers:
                if R2 < ST:
                    alpha = np.random.rand() + 0.01
                    decay = np.clip(-t / (alpha * self.max_iter), -50, 0)
                    self.positions[i] = self.positions[i] * np.exp(decay)
                else:
                    Q = np.random.randn(self.dim) * 0.1
                    self.positions[i] = self.positions[i] + Q

                # Potential field adjustment
                self.positions[i] += self._potential_force(self.positions[i], t)
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)

            # Update scroungers
            best_producer = self.positions[producers[0]]
            for i in scroungers:
                if np.random.rand() > 0.5:
                    # Follow best with noise
                    self.positions[i] = best_producer + np.random.randn(self.dim) * 0.1 * (self.ub - self.lb)
                else:
                    self.positions[i] = best_producer + np.abs(self.positions[i] - best_producer) * np.random.randn(self.dim) * 0.5

                self.positions[i] += self._potential_force(self.positions[i], t)
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)

            # Anti-predator behavior (danger awareness)
            num_aware = max(1, int(0.1 * self.num_sparrows))
            aware_idx = np.random.choice(self.num_sparrows, num_aware, replace=False)
            for i in aware_idx:
                if self.fitness[i] > self.best_fit:
                    beta = np.random.randn(self.dim) * 0.1
                    self.positions[i] = self.best_pos + beta * np.abs(self.positions[i] - self.best_pos)
                else:
                    K = np.random.uniform(-1, 1)
                    worst_fit = self.fitness[sorted_idx[-1]]
                    diff_fit = abs(self.fitness[i] - worst_fit) + 1e-10
                    self.positions[i] = self.positions[i] + K * (self.positions[i] - self.positions[sorted_idx[-1]]) / diff_fit * 0.01
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)

            # OBL jump every 10 iterations
            if (t + 1) % 10 == 0:
                self._opposition_based_learning()

            # Update fitness
            self.fitness = np.array([self.fitness_fn(p) for p in self.positions])
            min_idx = np.argmin(self.fitness)
            if self.fitness[min_idx] < self.best_fit:
                self.best_fit = self.fitness[min_idx]
                self.best_pos = self.positions[min_idx].copy()

            self.convergence.append(self.best_fit)

        return self.best_pos, self.best_fit


# ============== Run Benchmarks ==============

BENCHMARK_FUNCTIONS = {
    'Sphere':     (sphere_function,     (-100, 100), 0.0),
    'Rosenbrock': (rosenbrock_function, (-30, 30),   0.0),
    'Ackley':     (ackley_function,     (-32, 32),   0.0),
    'Rastrigin':  (rastrigin_function,  (-5.12, 5.12), 0.0),
    'Griewank':   (griewank_function,   (-600, 600), 0.0),
    'Schwefel':   (schwefel_function,   (-500, 500), 0.0),
}

VARIANTS = [
    ('Baseline',                   dict(use_chaotic_init=False, use_potential_field=False, use_obl=False)),
    ('Base + Chaotic Init',        dict(use_chaotic_init=True,  use_potential_field=False, use_obl=False)),
    ('Base + Potential Field',     dict(use_chaotic_init=False, use_potential_field=True,  use_obl=False)),
    ('Base + OBL',                 dict(use_chaotic_init=False, use_potential_field=False, use_obl=True)),
    ('All Combined',               dict(use_chaotic_init=True,  use_potential_field=True,  use_obl=True)),
]


def run_benchmarks(dim=30, num_sparrows=30, max_iter=500, runs=5):
    all_results = []

    for func_name, (func, bounds, optimal) in BENCHMARK_FUNCTIONS.items():
        print(f"\n{'='*60}")
        print(f"Function: {func_name} (optimal ≈ {optimal})")
        print('='*60)

        for variant_name, variant_flags in VARIANTS:
            fitnesses = []
            times = []
            for r in range(runs):
                ssa = BenchmarkSSA(
                    fitness_fn=func,
                    dim=dim,
                    bounds=bounds,
                    num_sparrows=num_sparrows,
                    max_iter=max_iter,
                    seed=42 + r,
                    **variant_flags
                )
                t0 = time.time()
                _, best_fit = ssa.optimize()
                dt = time.time() - t0
                fitnesses.append(best_fit)
                times.append(dt)

            mean_fit = np.mean(fitnesses)
            std_fit = np.std(fitnesses)
            mean_time = np.mean(times)

            all_results.append({
                'Function': func_name,
                'Variant': variant_name,
                'Mean': mean_fit,
                'Std': std_fit,
                'Best': np.min(fitnesses),
                'Worst': np.max(fitnesses),
                'Time_s': mean_time
            })

            print(f"  {variant_name:30s} | Mean: {mean_fit:12.4e} | Std: {std_fit:10.4e} | Best: {np.min(fitnesses):12.4e}")

    return all_results


def main():
    print("Running SSA Variant Benchmarks on Standard Test Functions")
    print("Dimensions: 30 | Sparrows: 30 | Iterations: 500 | Runs: 5\n")

    results = run_benchmarks(dim=30, num_sparrows=30, max_iter=500, runs=5)

    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = 'optimization_results/benchmark_results.csv'
    df.to_csv(csv_path, index=False)

    # Save to JSON
    with open('optimization_results/benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)

    # Pivot table: rows = functions, columns = variants
    pivot = df.pivot_table(index='Function', columns='Variant', values='Mean')
    print(pivot.to_string())

    # Compute improvement percentages relative to Baseline
    print("\n" + "="*80)
    print("IMPROVEMENT vs BASELINE (%)")
    print("="*80)

    improvement_data = []
    for func_name in BENCHMARK_FUNCTIONS.keys():
        func_df = df[df['Function'] == func_name]
        baseline_mean = func_df[func_df['Variant'] == 'Baseline']['Mean'].values[0]
        row = {'Function': func_name}
        for variant_name, _ in VARIANTS:
            var_mean = func_df[func_df['Variant'] == variant_name]['Mean'].values[0]
            if baseline_mean != 0:
                improvement = (baseline_mean - var_mean) / abs(baseline_mean) * 100
            else:
                improvement = 0 if var_mean == 0 else -100
            row[variant_name] = f"{improvement:+.2f}%"
        improvement_data.append(row)

    imp_df = pd.DataFrame(improvement_data)
    print(imp_df.to_string(index=False))

    print(f"\nResults saved to {csv_path} and optimization_results/benchmark_results.json")


if __name__ == '__main__':
    main()
