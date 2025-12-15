import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import time
import pandas as pd
import os

# Create a directory for results
if not os.path.exists("optimization_results"):
    os.makedirs("optimization_results")

# --- 1. BENCHMARK OBJECTIVE FUNCTIONS ---
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

# --- 2. ENHANCED ALGORITHM CLASS (COB-SSA) ---
class EnhancedSparrowSearchAlgorithm:
    def __init__(self, objective_function, num_sparrows=30, dimensions=30, max_iter=100, lb=-100, ub=100):
        self.objective_function = objective_function
        self.num_sparrows = num_sparrows
        self.dimensions = dimensions
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        
        self.positions = self.chaotic_initialization()
        self.fitness = np.apply_along_axis(self.objective_function, 1, self.positions)
        self.best_pos = self.positions[np.argmin(self.fitness)]
        self.best_fit = np.min(self.fitness)
        
        self.convergence_history = []
        self.trajectory_history = [] 
        self.producers = []
        self.scroungers = []

    def chaotic_initialization(self):
        """Improvement 1: Chaotic Initialization (Logistic Map)."""
        chaotic_pos = np.zeros((self.num_sparrows, self.dimensions))
        for d in range(self.dimensions):
            x = np.random.rand()
            for i in range(self.num_sparrows):
                x = 4.0 * x * (1 - x) 
                chaotic_pos[i, d] = self.lb + x * (self.ub - self.lb)
        return chaotic_pos

    def opposition_based_learning(self):
        """Improvement 2: Opposition-Based Learning (OBL)."""
        candidates_idx = np.random.choice(self.num_sparrows, max(1, int(0.1 * self.num_sparrows)), replace=False)
        for i in candidates_idx:
            current_x = self.positions[i]
            current_f = self.fitness[i]
            opposite_x = self.lb + self.ub - current_x
            opposite_x = np.clip(opposite_x, self.lb, self.ub)
            opposite_f = self.objective_function(opposite_x)
            if opposite_f < current_f:
                self.positions[i] = opposite_x
                self.fitness[i] = opposite_f

    def calculate_potential_forces(self, current_idx):
        """Improvement 4: Potential Field / Swarm Repulsion."""
        current_pos = self.positions[current_idx]
        force = np.zeros(self.dimensions)
        repulsion_radius = (self.ub - self.lb) * 0.05
        
        for i in range(self.num_sparrows):
            if i != current_idx:
                diff = current_pos - self.positions[i]
                dist = np.linalg.norm(diff)
                if dist < repulsion_radius and dist > 1e-10:
                    f_mag = 0.5 / (dist**2)
                    force += f_mag * (diff / dist)
        
        max_force = (self.ub - self.lb) * 0.05
        force_mag = np.linalg.norm(force)
        if force_mag > max_force:
            force = force / force_mag * max_force
        return force

    def update_roles(self):
        sorted_indices = np.argsort(self.fitness)
        num_producers = max(1, int(0.2 * self.num_sparrows))
        self.producers = sorted_indices[:num_producers]
        self.scroungers = sorted_indices[num_producers:]
    
    def update_producers(self, iter_num):
        R2 = np.random.rand()
        ST = 0.6
        w = 0.9 - 0.5 * (iter_num / self.max_iter) # Improvement 3: Adaptive Weight
        
        for i in self.producers:
            repulsion = self.calculate_potential_forces(i)
            if R2 < ST:
                self.positions[i] = self.positions[i] * np.exp(-i / (0.8 * self.max_iter)) + repulsion
            else:
                self.positions[i] += np.random.randn(self.dimensions) * w
            self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
    
    def update_scroungers(self, iter_num):
        w = 0.9 - 0.5 * (iter_num / self.max_iter)
        for i in self.scroungers:
            repulsion = self.calculate_potential_forces(i)
            if self.fitness[i] > np.median(self.fitness):
                self.positions[i] = w * self.positions[i] + \
                                    np.random.rand() * (self.best_pos - self.positions[i]) + \
                                    repulsion
            else:
                self.positions[i] += np.random.uniform(-0.1, 0.1, self.dimensions) * w
            self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
            
            new_fitness = self.objective_function(self.positions[i])
            if new_fitness < self.fitness[self.producers[-1]]:
                self.positions[self.producers[-1]] = self.positions[i].copy()
                self.fitness[self.producers[-1]] = new_fitness
    
    def anti_predator_escape(self):
        worst_index = np.argmax(self.fitness)
        worst_pos = self.positions[worst_index]
        for i in range(self.num_sparrows):
            if np.random.rand() < 0.1: 
                escape_factor = np.random.uniform(-1, 1, self.dimensions)
                self.positions[i] += escape_factor * (self.positions[i] - worst_pos) * 0.1
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
    
    def random_walk(self, t, max_t):
        r_t = np.random.rand(self.dimensions)
        r_t = np.where(r_t > 0.5, 1, 0)
        steps = 2 * r_t - 1
        X_t = np.cumsum(steps)
        progress_ratio = t / max_t
        a, b = np.min(self.positions, axis=0), np.max(self.positions, axis=0)
        c = self.best_pos - (1 - progress_ratio) * (b - a) * 0.5
        d = self.best_pos + (1 - progress_ratio) * (b - a) * 0.5
        X_normalized = (X_t - np.min(X_t)) * (d - c) / (np.max(X_t) - np.min(X_t) + 1e-10) + c
        X_normalized = np.clip(X_normalized, self.lb, self.ub)
        return X_normalized

    def optimize(self):
        for iter_num in range(self.max_iter):
            self.update_roles()
            
            # Save history for visualization (only first 2 dims)
            if self.dimensions >= 2:
                self.trajectory_history.append(self.positions[:, :2].copy())
            
            self.update_producers(iter_num)
            self.update_scroungers(iter_num)
            
            if np.random.rand() < 0.3:
                random_walk_pos = self.random_walk(iter_num, self.max_iter)
                random_walk_fitness = self.objective_function(random_walk_pos)
                if random_walk_fitness < self.best_fit:
                    self.best_pos = random_walk_pos
                    self.best_fit = random_walk_fitness
                    self.positions[np.argmax(self.fitness)] = random_walk_pos
            
            self.anti_predator_escape()
            if iter_num % 5 == 0: self.opposition_based_learning()
            
            self.fitness = np.apply_along_axis(self.objective_function, 1, self.positions)
            best_index = np.argmin(self.fitness)
            if self.fitness[best_index] < self.best_fit:
                self.best_fit = self.fitness[best_index]
                self.best_pos = self.positions[best_index].copy()
            
            self.convergence_history.append(self.best_fit)
        
        return self.best_pos, self.best_fit

# --- 3. PLOTTING HELPERS ---

def plot_separate_convergence(name, curve):
    """Saves individual convergence curve for a function."""
    plt.figure(figsize=(8, 6))
    plt.semilogy(range(1, len(curve)+1), curve, linewidth=2, color='darkblue')
    plt.title(f"{name} Function - Convergence Curve", fontsize=14)
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Best Fitness (Log Scale)", fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    filename = f"optimization_results/Convergence_{name.replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

def plot_trajectory_2d(name, func, lb, ub):
    """Runs a quick 2D simulation to plot particle trajectory."""
    algo = EnhancedSparrowSearchAlgorithm(func, num_sparrows=30, dimensions=2, max_iter=50, lb=lb, ub=ub)
    algo.optimize()
    
    x = np.linspace(lb, ub, 100)
    y = np.linspace(lb, ub, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
            
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.6)
    
    history = np.array(algo.trajectory_history) 
    plt.scatter(history[0, :, 0], history[0, :, 1], c='white', edgecolor='black', s=30, label='Start')
    plt.scatter(history[-1, :, 0], history[-1, :, 1], c='red', marker='x', s=50, label='End')
    
    plt.title(f"{name} - Swarm Trajectory (2D)", fontsize=14)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.tight_layout()
    filename = f"optimization_results/Trajectory_{name.replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

def plot_3d_landscape(name, func, lb, ub):
    """Saves a 3D surface plot of the function."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.linspace(lb, ub, 50)
    y = np.linspace(lb, ub, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for r in range(X.shape[0]):
        for c in range(X.shape[1]):
            pt = np.array([X[r,c], Y[r,c]])
            try:
                Z[r,c] = func(pt)
            except:
                Z[r,c] = 0 
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    ax.set_title(f"{name} Function Landscape", fontsize=16)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x)')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    filename = f"optimization_results/3D_Landscape_{name.replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

def plot_boxplots(results_map):
    """Plots stability boxplots for all functions."""
    data = []
    labels = []
    for name, scores in results_map.items():
        data.append(scores)
        labels.append(name)
        
    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=labels)
    plt.yscale('log') 
    plt.title("Fitness Distribution (Stability Analysis)", fontsize=14)
    plt.ylabel("Best Fitness (Log Scale)")
    plt.grid(True, which="both", alpha=0.2)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("optimization_results/Boxplot_Stability.png")
    plt.close()
    print("Saved: Boxplot_Stability.png")

# --- 4. MAIN VALIDATION SUITE ---

def run_comprehensive_validation():
    runs_per_func = 10 
    dim = 30           
    pop_size = 50
    iterations = 200
    
    benchmarks = {
        "Sphere": {"func": sphere_function, "lb": -100, "ub": 100},
        "Rosenbrock": {"func": rosenbrock_function, "lb": -30, "ub": 30},
        "Ackley": {"func": ackley_function, "lb": -32, "ub": 32},
        "Rastrigin": {"func": rastrigin_function, "lb": -5.12, "ub": 5.12},
        "Griewank": {"func": griewank_function, "lb": -600, "ub": 600},
        "Schwefel": {"func": schwefel_function, "lb": -500, "ub": 500},
    }
    
    results_table = []
    all_final_scores = {} 
    
    print(f"Starting Validation Suite: {runs_per_func} runs per function, {dim} dimensions.\n")
    
    for name, params in benchmarks.items():
        print(f"Benchmarking {name} Function...")
        
        best_scores = []
        times = []
        avg_curve = np.zeros(iterations)
        
        # 1. Run Statistics
        for run in range(runs_per_func):
            start_time = time.time()
            algo = EnhancedSparrowSearchAlgorithm(
                params["func"], pop_size, dim, iterations, params["lb"], params["ub"]
            )
            _, best_val = algo.optimize()
            end_time = time.time()
            
            best_scores.append(best_val)
            times.append(end_time - start_time)
            
            curve = np.array(algo.convergence_history[:iterations])
            if len(curve) < iterations:
                curve = np.pad(curve, (0, iterations - len(curve)), 'edge')
            avg_curve += curve
            
        avg_curve /= runs_per_func
        all_final_scores[name] = best_scores
        
        # 2. Save Plots (Convergence, Trajectory, 3D)
        plot_separate_convergence(name, avg_curve)
        plot_trajectory_2d(name, params["func"], params["lb"], params["ub"])
        plot_3d_landscape(name, params["func"], params["lb"], params["ub"])
        
        # 3. Record Statistics
        results_table.append({
            "Function": name,
            "Best": f"{np.min(best_scores):.4e}",
            "Worst": f"{np.max(best_scores):.4e}",
            "Mean": f"{np.mean(best_scores):.4e}",
            "Std Dev": f"{np.std(best_scores):.4e}",
            "Avg Time (s)": f"{np.mean(times):.4f}"
        })

    # 4. Save Global Statistics
    df = pd.DataFrame(results_table)
    print("\n" + "="*80)
    print("TABLE II: COMPARISON OF OPTIMIZATION RESULTS (P-COB-SSA)")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    df.to_csv("optimization_results/Table_II_Results.csv", index=False)
    
    # 5. Save Boxplot
    plot_boxplots(all_final_scores)
    
    print("\nAll results, images, and tables saved to 'optimization_results' folder.")

if __name__ == "__main__":
    run_comprehensive_validation()