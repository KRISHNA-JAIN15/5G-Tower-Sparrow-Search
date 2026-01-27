"""
Enhanced Benchmark SSA variants on Standard + Harder Test Functions
with improved Chaotic Init, Potential Field, and OBL implementations.

Includes:
- Wilcoxon Rank-Sum Test for statistical significance
- Search Trajectory Tortuosity (τ) analysis
"""
import sys
import time
import math
import numpy as np
import pandas as pd
import json
import warnings
from scipy import stats  # For Wilcoxon test
warnings.filterwarnings('ignore')

# Fix for np.math usage
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

# ============== BENCHMARK FUNCTIONS ==============

# --- Primary Test Functions (5 functions for statistical analysis) ---

def sphere_function(x):
    """Unimodal, convex - Global min at origin = 0 (F5)"""
    return np.sum(x ** 2)

def schwefel_function(x):
    """Deceptive multimodal - Global min ≈ 0 at x_i ≈ 420.9687 (F1)"""
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def lunacek_bi_rastrigin_function(x):
    """
    Lunacek Bi-Rastrigin Function - Two funnels, very deceptive (F2)
    Global min ≈ 0
    """
    n = len(x)
    d = 1
    s = 1 - 1 / (2 * np.sqrt(n + 20) - 8.2)
    mu1 = 2.5
    mu2 = -np.sqrt((mu1**2 - d) / s)
    
    sum1 = np.sum((x - mu1)**2)
    sum2 = d * n + s * np.sum((x - mu2)**2)
    sum3 = 10 * (n - np.sum(np.cos(2 * np.pi * (x - mu1))))
    
    return min(sum1, sum2) + sum3

def styblinski_tang_function(x):
    """Multimodal - Global min ≈ -39.16599 * d at x_i ≈ -2.903534 (F3)"""
    return 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x)

def qing_function(x):
    """
    Qing Function - Multimodal with multiple global minima (F4)
    Global min = 0 at x_i = ±sqrt(i)
    """
    i = np.arange(1, len(x) + 1)
    return np.sum((x**2 - i)**2)


# --- COMMENTED OUT: Other benchmark functions (kept for reference) ---
"""
def rosenbrock_function(x):
    # Unimodal, non-convex valley - Global min at (1,1,...,1) = 0
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

def ackley_function(x):
    # Multimodal - Global min at origin = 0
    a, b, c = 20, 0.2, 2 * np.pi
    d = len(x)
    return -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / d)) - np.exp(np.sum(np.cos(c * x)) / d) + a + np.exp(1)

def rastrigin_function(x):
    # Highly multimodal - Global min at origin = 0
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def griewank_function(x):
    # Multimodal with many local minima - Global min at origin = 0
    sum_sq = np.sum(x**2) / 4000
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_sq - prod_cos + 1

def levy_function(x):
    # Complex multimodal - Global min at (1,1,...,1) = 0
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0]) ** 2
    term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
    term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
    return term1 + term2 + term3

def michalewicz_function(x, m=10):
    # Very steep valleys - Global min depends on dimension
    i = np.arange(1, len(x) + 1)
    return -np.sum(np.sin(x) * np.sin(i * x**2 / np.pi) ** (2 * m))

def zakharov_function(x):
    # Unimodal but hard to converge - Global min at origin = 0
    i = np.arange(1, len(x) + 1)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(0.5 * i * x)
    return sum1 + sum2**2 + sum2**4

def dixon_price_function(x):
    # Valley function - Global min = 0
    term1 = (x[0] - 1) ** 2
    i = np.arange(2, len(x) + 1)
    term2 = np.sum(i * (2 * x[1:] ** 2 - x[:-1]) ** 2)
    return term1 + term2

def drop_wave_function(x):
    # Highly multimodal with many local minima - Global min = -1 at origin
    sum_sq = np.sum(x ** 2)
    return -(1 + np.cos(12 * np.sqrt(sum_sq))) / (0.5 * sum_sq + 2)

def alpine_function(x):
    # Alpine N.1 Function - Many local minima, Global min at origin = 0
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x))

def salomon_function(x):
    # Salomon Function - Multimodal with circular ridges, Global min at origin = 0
    sum_sq = np.sqrt(np.sum(x**2))
    return 1 - np.cos(2 * np.pi * sum_sq) + 0.1 * sum_sq

def schaffer_f7_function(x):
    # Schaffer F7 Function - Highly multimodal, Global min at origin = 0
    n = len(x)
    s = np.sqrt(x[:-1]**2 + x[1:]**2)
    return (np.sum(np.sqrt(s) * (np.sin(50 * s**0.2) + 1)) / (n - 1))**2

def happy_cat_function(x):
    # Happy Cat Function - Deceptive with narrow global basin
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_x = np.sum(x)
    return ((sum_sq - n)**2)**0.125 + (0.5 * sum_sq + sum_x) / n + 0.5

def periodic_function(x):
    # Periodic Function - Highly oscillatory, Global min at origin = 0.9
    return 1 + np.sum(np.sin(x)**2) - 0.1 * np.exp(-np.sum(x**2))

def xin_she_yang_n4_function(x):
    # Xin-She Yang N.4 Function - Highly irregular, Global min at origin = -1
    sum1 = np.sum(np.sin(x)**2)
    sum2 = np.sum(x**2)
    return (sum1 - np.exp(-sum2)) * np.exp(-np.sum(np.sin(np.sqrt(np.abs(x)))**2))

def weierstrass_function(x, a=0.5, b=3, k_max=20):
    # Weierstrass Function - Continuous everywhere but differentiable nowhere
    n = len(x)
    result = 0
    for i in range(n):
        for k in range(k_max + 1):
            result += a**k * np.cos(2 * np.pi * b**k * (x[i] + 0.5))
    constant = 0
    for k in range(k_max + 1):
        constant += a**k * np.cos(np.pi * b**k)
    return result - n * constant

def bent_cigar_function(x):
    # Bent Cigar Function - Unimodal but ill-conditioned, Global min at origin = 0
    return x[0]**2 + 1e6 * np.sum(x[1:]**2)

def katsuura_function(x):
    # Katsuura Function - Continuous but highly irregular
    n = len(x)
    result = 1.0
    for i in range(n):
        inner_sum = 0
        for j in range(1, 33):
            inner_sum += np.abs(2**j * x[i] - round(2**j * x[i])) / 2**j
        result *= (1 + (i + 1) * inner_sum)**(10.0 / n**1.2)
    return (10.0 / n**2) * result - (10.0 / n**2)
"""


# ============== ENHANCED SSA IMPLEMENTATIONS ==============

class EnhancedBenchmarkSSA:
    """
    Enhanced SSA with improved implementations of:
    1. Chaotic Initialization (Tent + Logistic hybrid)
    2. Adaptive Potential Field Navigation
    3. Enhanced Opposition-Based Learning with Lévy flights
    """
    def __init__(self, fitness_fn, dim=30, bounds=(-100, 100),
                 num_sparrows=50, max_iter=500, seed=None,
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
        
        # Track history for adaptive mechanisms
        self.fitness_history = [self.best_fit]
        self.stagnation_count = 0
        self.convergence = []
        
        # Track trajectory for tortuosity calculation
        self.trajectory = [self.best_pos.copy()]  # Store best position at each iteration
        self.start_pos = self.best_pos.copy()  # Initial best position

    def _tent_map(self, x):
        """Tent chaotic map for better uniformity"""
        if x < 0.5:
            return 2 * x
        else:
            return 2 * (1 - x)
    
    def _logistic_map(self, x):
        """Logistic chaotic map"""
        return 4.0 * x * (1 - x)
    
    def _sinusoidal_map(self, x):
        """Sinusoidal chaotic map for diversity"""
        return np.sin(np.pi * x)

    def _initialize_positions(self):
        """
        IMPROVED CHAOTIC INITIALIZATION:
        Uses hybrid Tent-Logistic-Sinusoidal maps for better coverage
        """
        if not self.use_chaotic_init:
            return np.random.uniform(self.lb, self.ub, (self.num_sparrows, self.dim))
        
        positions = np.zeros((self.num_sparrows, self.dim))
        
        # Use different chaotic maps for different segments
        x_tent = np.random.rand()
        x_logistic = np.random.rand()
        x_sin = np.random.rand()
        
        for i in range(self.num_sparrows):
            for j in range(self.dim):
                # Hybrid: combine multiple chaotic maps
                x_tent = self._tent_map(x_tent)
                x_logistic = self._logistic_map(x_logistic)
                x_sin = abs(self._sinusoidal_map(x_sin))
                
                # Weighted combination for better distribution
                if i % 3 == 0:
                    chaos_val = x_tent
                elif i % 3 == 1:
                    chaos_val = x_logistic
                else:
                    chaos_val = 0.5 * x_tent + 0.3 * x_logistic + 0.2 * x_sin
                
                positions[i, j] = self.lb + chaos_val * (self.ub - self.lb)
        
        return positions

    def _adaptive_potential_force(self, pos, t):
        """
        IMPROVED POTENTIAL FIELD NAVIGATION:
        - Adaptive attraction strength based on progress
        - Multi-attractor system with best positions
        - Boundary repulsion to avoid stagnation at edges
        """
        if not self.use_potential_field:
            return np.zeros(self.dim)
        
        progress = t / self.max_iter
        force = np.zeros(self.dim)
        
        # 1. Attraction to global best (increasing with progress)
        attract_strength = 0.3 + 0.4 * progress  # 0.3 -> 0.7
        attract = attract_strength * (self.best_pos - pos)
        
        # 2. Attraction to top-k elite positions (for diversity)
        if t > 10:
            k = max(3, int(0.1 * self.num_sparrows))
            elite_idx = np.argsort(self.fitness)[:k]
            elite_center = np.mean(self.positions[elite_idx], axis=0)
            elite_attract = 0.1 * (1 - progress) * (elite_center - pos)
            attract += elite_attract
        
        # 3. Boundary repulsion (soft constraint)
        boundary_force = np.zeros(self.dim)
        margin = 0.1 * (self.ub - self.lb)
        for d in range(self.dim):
            if pos[d] < self.lb + margin:
                boundary_force[d] = 0.5 * (self.lb + margin - pos[d])
            elif pos[d] > self.ub - margin:
                boundary_force[d] = 0.5 * (self.ub - margin - pos[d])
        
        # 4. Stagnation escape force (random perturbation when stuck)
        escape_force = np.zeros(self.dim)
        if self.stagnation_count > 20:
            escape_strength = min(0.5, 0.1 * (self.stagnation_count - 20))
            escape_force = escape_strength * np.random.randn(self.dim)
        
        force = attract + boundary_force + escape_force
        
        # Clip force magnitude
        max_force = 0.5 * (self.ub - self.lb) * (1 - 0.5 * progress)
        force_mag = np.linalg.norm(force)
        if force_mag > max_force:
            force = force / force_mag * max_force
        
        return force

    def _levy_flight(self, beta=1.5):
        """Generate Lévy flight step for enhanced exploration"""
        sigma_u = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.randn(self.dim) * sigma_u
        v = np.random.randn(self.dim)
        step = u / (np.abs(v) ** (1 / beta))
        return step

    def _enhanced_opposition_learning(self, t):
        """
        IMPROVED OPPOSITION-BASED LEARNING:
        - Generalized opposition with random factor
        - Lévy-flight enhanced opposition
        - Selective application based on fitness ranking
        """
        if not self.use_obl:
            return
        
        progress = t / self.max_iter
        
        # Apply OBL to bottom 30% performers more frequently
        sorted_idx = np.argsort(self.fitness)
        num_candidates = max(2, int(0.3 * self.num_sparrows))
        
        # Select from worse performers
        candidate_pool = sorted_idx[-num_candidates:]
        candidates = np.random.choice(candidate_pool, max(1, int(0.5 * num_candidates)), replace=False)
        
        for i in candidates:
            original_pos = self.positions[i].copy()
            original_fit = self.fitness[i]
            
            # Generalized quasi-opposition with random factor
            k = np.random.uniform(0.5, 1.0)  # Quasi-opposition factor
            center = (self.lb + self.ub) / 2
            
            # Basic opposition
            opposite_pos = self.lb + self.ub - original_pos
            
            # Generalized quasi-opposition
            quasi_opposite = center + k * (center - original_pos)
            
            # Lévy-enhanced opposition (exploration boost)
            levy_step = self._levy_flight() * 0.1 * (self.ub - self.lb) * (1 - progress)
            levy_opposite = opposite_pos + levy_step
            
            # Choose best among three candidates
            candidates_pos = [opposite_pos, quasi_opposite, levy_opposite]
            candidates_pos = [np.clip(p, self.lb, self.ub) for p in candidates_pos]
            candidates_fit = [self.fitness_fn(p) for p in candidates_pos]
            
            best_candidate_idx = np.argmin(candidates_fit)
            best_candidate_pos = candidates_pos[best_candidate_idx]
            best_candidate_fit = candidates_fit[best_candidate_idx]
            
            # Accept if better
            if best_candidate_fit < original_fit:
                self.positions[i] = best_candidate_pos
                self.fitness[i] = best_candidate_fit

    def optimize(self):
        ST = 0.8  # Safety threshold
        PD = 0.2  # Proportion of producers

        for t in range(self.max_iter):
            prev_best = self.best_fit
            
            sorted_idx = np.argsort(self.fitness)
            num_producers = max(1, int(PD * self.num_sparrows))
            producers = sorted_idx[:num_producers]
            scroungers = sorted_idx[num_producers:]

            R2 = np.random.rand()
            progress = t / self.max_iter
            
            # Update producers
            for i in producers:
                if R2 < ST:
                    # Safe foraging with adaptive decay
                    alpha = np.random.rand() + 0.01
                    decay = np.clip(-t / (alpha * self.max_iter), -30, 0)
                    new_pos = self.positions[i] * np.exp(decay)
                else:
                    # Danger detected - random movement
                    Q = np.random.randn(self.dim) * 0.1 * (1 - progress)
                    new_pos = self.positions[i] + Q

                # Apply potential field
                new_pos += self._adaptive_potential_force(new_pos, t)
                self.positions[i] = np.clip(new_pos, self.lb, self.ub)

            # Update scroungers
            best_producer = self.positions[producers[0]]
            for i in scroungers:
                if np.random.rand() > 0.5:
                    # Follow best producer with noise
                    A = np.random.randn(self.dim)
                    scale = 0.3 * (1 - progress) + 0.05
                    new_pos = best_producer + scale * A * (self.ub - self.lb)
                else:
                    # Random walk toward best
                    new_pos = self.positions[i] + np.random.randn(self.dim) * 0.2 * np.abs(self.positions[i] - best_producer)

                new_pos += self._adaptive_potential_force(new_pos, t) * 0.5
                self.positions[i] = np.clip(new_pos, self.lb, self.ub)

            # Anti-predator behavior
            num_aware = max(1, int(0.1 * self.num_sparrows))
            aware_idx = np.random.choice(self.num_sparrows, num_aware, replace=False)
            for i in aware_idx:
                if self.fitness[i] > self.best_fit:
                    beta = np.random.randn(self.dim) * 0.1
                    new_pos = self.best_pos + beta * np.abs(self.positions[i] - self.best_pos)
                else:
                    K = np.random.uniform(-1, 1)
                    worst_idx = sorted_idx[-1]
                    diff = self.positions[i] - self.positions[worst_idx]
                    new_pos = self.positions[i] + K * diff * 0.01
                self.positions[i] = np.clip(new_pos, self.lb, self.ub)

            # Enhanced OBL every 5 iterations
            if (t + 1) % 5 == 0:
                self._enhanced_opposition_learning(t)

            # Update fitness
            self.fitness = np.array([self.fitness_fn(p) for p in self.positions])
            
            # Update global best
            min_idx = np.argmin(self.fitness)
            if self.fitness[min_idx] < self.best_fit:
                self.best_fit = self.fitness[min_idx]
                self.best_pos = self.positions[min_idx].copy()
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1

            self.convergence.append(self.best_fit)
            self.fitness_history.append(self.best_fit)
            self.trajectory.append(self.best_pos.copy())  # Track trajectory

        return self.best_pos, self.best_fit
    
    def compute_tortuosity(self):
        """
        Compute Search Trajectory Tortuosity (τ)
        τ = Total Path Length / Net Euclidean Displacement
        
        τ → 1.0: Ballistic efficiency (direct path)
        τ >> 1.0: Erratic Brownian motion (inefficient wandering)
        """
        if len(self.trajectory) < 2:
            return 1.0
        
        # Total path length (sum of step-wise distances)
        total_path_length = 0.0
        for t in range(1, len(self.trajectory)):
            step_dist = np.linalg.norm(
                np.array(self.trajectory[t]) - np.array(self.trajectory[t-1])
            )
            total_path_length += step_dist
        
        # Net displacement (start to final best position)
        net_displacement = np.linalg.norm(
            np.array(self.trajectory[-1]) - np.array(self.trajectory[0])
        )
        
        # Avoid division by zero
        if net_displacement < 1e-10:
            # If no net movement, return high tortuosity if path was long
            return total_path_length if total_path_length > 0 else 1.0
        
        tortuosity = total_path_length / net_displacement
        return tortuosity


# ============== BENCHMARK CONFIGURATION ==============

# Only 5 functions for statistical analysis (as per paper Table III)
BENCHMARK_FUNCTIONS = {
    'Schwefel (F1)':   (schwefel_function,              (-500, 500),   0.0),
    'Lunacek (F2)':    (lunacek_bi_rastrigin_function,  (-5, 5),       0.0),
    'Styblinski (F3)': (styblinski_tang_function,       (-5, 5),       None),  # -39.16*d
    'Qing (F4)':       (qing_function,                  (-500, 500),   0.0),
    'Sphere (F5)':     (sphere_function,                (-100, 100),   0.0),
}

# COMMENTED OUT: Other functions (kept for reference)
"""
BENCHMARK_FUNCTIONS_FULL = {
    'Sphere':       (sphere_function,         (-100, 100),    0.0),
    'Rosenbrock':   (rosenbrock_function,     (-30, 30),      0.0),
    'Ackley':       (ackley_function,         (-32, 32),      0.0),
    'Rastrigin':    (rastrigin_function,      (-5.12, 5.12),  0.0),
    'Griewank':     (griewank_function,       (-600, 600),    0.0),
    'Schwefel':     (schwefel_function,       (-500, 500),    0.0),
    'Levy':         (levy_function,           (-10, 10),      0.0),
    'Zakharov':     (zakharov_function,       (-10, 10),      0.0),
    'Alpine':       (alpine_function,         (-10, 10),      0.0),
    'Salomon':      (salomon_function,        (-100, 100),    0.0),
    'Michalewicz':  (michalewicz_function,    (0, np.pi),     None),
    'Dixon-Price':  (dixon_price_function,    (-10, 10),      0.0),
    'Styblinski':   (styblinski_tang_function,(-5, 5),        None),
    'Schaffer-F7':  (schaffer_f7_function,    (-100, 100),    0.0),
    'Happy-Cat':    (happy_cat_function,      (-2, 2),        0.0),
    'Qing':         (qing_function,           (-500, 500),    0.0),
    'Lunacek':      (lunacek_bi_rastrigin_function, (-5, 5),  0.0),
    'Bent-Cigar':   (bent_cigar_function,     (-100, 100),    0.0),
}
"""

VARIANTS = [
    ('Baseline',                   dict(use_chaotic_init=False, use_potential_field=False, use_obl=False)),
    ('Chaotic Init',               dict(use_chaotic_init=True,  use_potential_field=False, use_obl=False)),
    ('Potential Field',            dict(use_chaotic_init=False, use_potential_field=True,  use_obl=False)),
    ('OBL',                        dict(use_chaotic_init=False, use_potential_field=False, use_obl=True)),
    ('Chaotic + PF',               dict(use_chaotic_init=True,  use_potential_field=True,  use_obl=False)),
    ('Chaotic + OBL',              dict(use_chaotic_init=True,  use_potential_field=False, use_obl=True)),
    ('PF + OBL',                   dict(use_chaotic_init=False, use_potential_field=True,  use_obl=True)),
    ('All Combined (COB-SSA)',     dict(use_chaotic_init=True,  use_potential_field=True,  use_obl=True)),
]


def run_benchmarks(dim=30, num_sparrows=50, max_iter=500, runs=10):
    all_results = []
    # Store raw fitness values for statistical tests
    raw_fitness_data = {}  # {func_name: {variant_name: [fitness_values]}}
    # Store tortuosity values
    tortuosity_data = {}   # {func_name: {variant_name: [tortuosity_values]}}

    for func_name, (func, bounds, optimal) in BENCHMARK_FUNCTIONS.items():
        print(f"\n{'='*70}")
        print(f"Function: {func_name} | Bounds: {bounds} | Optimal: {optimal}")
        print('='*70)
        
        raw_fitness_data[func_name] = {}
        tortuosity_data[func_name] = {}

        for variant_name, variant_flags in VARIANTS:
            fitnesses = []
            times = []
            tortuosities = []
            
            for r in range(runs):
                ssa = EnhancedBenchmarkSSA(
                    fitness_fn=func,
                    dim=dim,
                    bounds=bounds,
                    num_sparrows=num_sparrows,
                    max_iter=max_iter,
                    seed=42 + r * 7,  # Different seeds
                    **variant_flags
                )
                t0 = time.time()
                _, best_fit = ssa.optimize()
                dt = time.time() - t0
                fitnesses.append(best_fit)
                times.append(dt)
                tortuosities.append(ssa.compute_tortuosity())
            
            # Store raw data for statistical tests
            raw_fitness_data[func_name][variant_name] = fitnesses
            tortuosity_data[func_name][variant_name] = tortuosities

            mean_fit = np.mean(fitnesses)
            std_fit = np.std(fitnesses)
            best_fit = np.min(fitnesses)
            worst_fit = np.max(fitnesses)
            mean_time = np.mean(times)
            mean_tortuosity = np.mean(tortuosities)

            all_results.append({
                'Function': func_name,
                'Variant': variant_name,
                'Mean': mean_fit,
                'Std': std_fit,
                'Best': best_fit,
                'Worst': worst_fit,
                'Time_s': mean_time,
                'Tortuosity': mean_tortuosity
            })

            print(f"  {variant_name:25s} | Mean: {mean_fit:14.6e} | Std: {std_fit:12.4e} | Best: {best_fit:14.6e} | τ: {mean_tortuosity:.4f}")

    return all_results, raw_fitness_data, tortuosity_data


def wilcoxon_rank_sum_test(raw_fitness_data, alpha=0.05):
    """
    Perform Wilcoxon Rank-Sum Test (Mann-Whitney U Test) 
    to assess statistical significance of COB-SSA vs Baseline.
    
    H0: No difference between Baseline and COB-SSA
    H1: COB-SSA performs differently from Baseline
    
    Returns DataFrame with p-values and statistical verdicts.
    """
    print("\n" + "="*70)
    print("TABLE III: Wilcoxon Rank-Sum Test Results (p-values vs. Baseline)")
    print("α = 0.05")
    print("="*70)
    print(f"{'Function':20s} | {'p-value':15s} | {'Statistical Verdict':25s}")
    print("-"*70)
    
    results = []
    
    for func_name, variant_data in raw_fitness_data.items():
        baseline_fitness = variant_data.get('Baseline', [])
        cob_ssa_fitness = variant_data.get('All Combined (COB-SSA)', [])
        
        if len(baseline_fitness) > 0 and len(cob_ssa_fitness) > 0:
            # Perform Wilcoxon Rank-Sum Test (Mann-Whitney U)
            statistic, p_value = stats.mannwhitneyu(
                baseline_fitness, 
                cob_ssa_fitness, 
                alternative='two-sided'
            )
            
            # Determine statistical verdict
            if p_value < alpha:
                verdict = "Significant (✓)"
            else:
                verdict = "Insignificant (≈)"
            
            results.append({
                'Function': func_name,
                'p-value': p_value,
                'Verdict': verdict
            })
            
            print(f"{func_name:20s} | {p_value:<15.2e} | {verdict:25s}")
    
    print("="*70)
    return pd.DataFrame(results)


def analyze_tortuosity(tortuosity_data):
    """
    Analyze Search Trajectory Tortuosity (τ) for each function and variant.
    
    τ → 1.0: Ballistic efficiency (direct, efficient path)
    τ >> 1.0: Erratic Brownian motion (inefficient wandering)
    """
    print("\n" + "="*70)
    print("TABLE IV: Search Trajectory Tortuosity (τ) Analysis")
    print("τ → 1.0: Ballistic efficiency | τ >> 1.0: Erratic motion")
    print("="*70)
    
    results = []
    
    for func_name, variant_data in tortuosity_data.items():
        print(f"\n{func_name}:")
        print("-"*50)
        
        for variant_name, tortuosities in variant_data.items():
            mean_tau = np.mean(tortuosities)
            std_tau = np.std(tortuosities)
            
            # Classify efficiency
            if mean_tau < 2.0:
                efficiency = "High (Ballistic)"
            elif mean_tau < 5.0:
                efficiency = "Medium"
            elif mean_tau < 10.0:
                efficiency = "Low"
            else:
                efficiency = "Very Low (Erratic)"
            
            results.append({
                'Function': func_name,
                'Variant': variant_name,
                'Mean_τ': mean_tau,
                'Std_τ': std_tau,
                'Efficiency': efficiency
            })
            
            print(f"  {variant_name:25s} | τ = {mean_tau:8.4f} ± {std_tau:6.4f} | {efficiency}")
    
    print("\n" + "="*70)
    return pd.DataFrame(results)


def print_improvement_table(df):
    """Print improvement percentages vs baseline"""
    print("\n" + "="*90)
    print("IMPROVEMENT vs BASELINE (%) - Negative means better (minimization)")
    print("="*90)
    
    funcs = df['Function'].unique()
    variants = [v[0] for v in VARIANTS if v[0] != 'Baseline']
    
    # Header
    header = f"{'Function':15s}"
    for v in variants:
        header += f" | {v[:12]:>12s}"
    print(header)
    print("-" * len(header))
    
    for func_name in funcs:
        func_df = df[df['Function'] == func_name]
        baseline = func_df[func_df['Variant'] == 'Baseline']['Mean'].values[0]
        
        row = f"{func_name:15s}"
        for variant_name in variants:
            var_mean = func_df[func_df['Variant'] == variant_name]['Mean'].values[0]
            if baseline != 0:
                improvement = (baseline - var_mean) / abs(baseline) * 100
            else:
                improvement = 0 if var_mean == 0 else -100
            row += f" | {improvement:+11.2f}%"
        print(row)


def main():
    print("="*70)
    print("ENHANCED SSA VARIANTS BENCHMARK")
    print("With Statistical Significance & Trajectory Tortuosity Analysis")
    print("Dimensions: 30 | Sparrows: 50 | Iterations: 500 | Runs: 10")
    print("="*70)

    results, raw_fitness_data, tortuosity_data = run_benchmarks(
        dim=30, num_sparrows=50, max_iter=500, runs=10
    )

    # Save to CSV and JSON
    df = pd.DataFrame(results)
    csv_path = 'optimization_results/benchmark_v2_results.csv'
    df.to_csv(csv_path, index=False)
    
    with open('optimization_results/benchmark_v2_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY TABLE (Mean Fitness)")
    print("="*70)
    
    pivot = df.pivot_table(index='Function', columns='Variant', values='Mean')
    print(pivot.to_string())
    
    # Print improvement table
    print_improvement_table(df)
    
    # ============== STATISTICAL ANALYSIS ==============
    
    # 1. Wilcoxon Rank-Sum Test (Table III)
    wilcoxon_df = wilcoxon_rank_sum_test(raw_fitness_data, alpha=0.05)
    wilcoxon_df.to_csv('optimization_results/wilcoxon_test_results.csv', index=False)
    
    # 2. Tortuosity Analysis (Table IV)
    tortuosity_df = analyze_tortuosity(tortuosity_data)
    tortuosity_df.to_csv('optimization_results/tortuosity_results.csv', index=False)
    
    # Best variant per function
    print("\n" + "="*70)
    print("BEST VARIANT PER FUNCTION")
    print("="*70)
    
    for func_name in df['Function'].unique():
        func_df = df[df['Function'] == func_name]
        best_row = func_df.loc[func_df['Mean'].idxmin()]
        baseline = func_df[func_df['Variant'] == 'Baseline']['Mean'].values[0]
        improvement = (baseline - best_row['Mean']) / abs(baseline) * 100 if baseline != 0 else 0
        print(f"{func_name:20s} -> {best_row['Variant']:25s} (Mean: {best_row['Mean']:.6e}, Improvement: {improvement:+.2f}%)")

    print(f"\nResults saved to {csv_path}")
    print("Statistical test results saved to optimization_results/wilcoxon_test_results.csv")
    print("Tortuosity analysis saved to optimization_results/tortuosity_results.csv")


if __name__ == '__main__':
    main()
