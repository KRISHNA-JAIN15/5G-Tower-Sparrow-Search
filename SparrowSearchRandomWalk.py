import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# --- Objective Functions ---
def sphere_function(x):
    return np.sum(x ** 2)

def rosenbrock_function(x):
    return sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

def ackley_function(x):
    a, b, c = 20, 0.2, 2 * np.pi
    d = len(x)
    return -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / d)) - np.exp(np.sum(np.cos(c * x)) / d) + a + np.exp(1)

def weighted_quadratic(x):
    a = np.linspace(1, 10, len(x))
    return np.sum(a * x**2)

def beale_variant(x):
    # Modified to handle dimensions > 2 by taking first two
    x0, x1 = x[0], x[1]
    term1 = (1.5 - x0 + x0 * x1)**2
    term2 = (2.25 - x0 + x0 * x1**2)**2
    return term1 + term2 + np.sum(x[2:]**2) # Penalty for extra dims

def schwefel_function(x):
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

class EnhancedSparrowSearchAlgorithm:
    def __init__(self, objective_function, num_sparrows=30, dimensions=3, max_iter=100, lb=-3, ub=3):
        self.objective_function = objective_function
        self.num_sparrows = num_sparrows
        self.dimensions = dimensions
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        
        # IMPROVEMENT 1: Chaotic Initialization
        # Use Logistic Map for better initial coverage
        self.positions = self.chaotic_initialization()
        
        self.fitness = np.apply_along_axis(self.objective_function, 1, self.positions)
        self.best_pos = self.positions[np.argmin(self.fitness)]
        self.best_fit = np.min(self.fitness)
        
        self.history = []
        self.producers_count = []
        self.scroungers_count = []
        self.convergence_history = []
        self.producers = []
        self.scroungers = []

    def chaotic_initialization(self):
        """
        Initialize positions using Logistic Chaotic Map.
        x_k+1 = 4 * x_k * (1 - x_k)
        """
        chaotic_pos = np.zeros((self.num_sparrows, self.dimensions))
        for d in range(self.dimensions):
            x = np.random.rand()
            for i in range(self.num_sparrows):
                x = 4.0 * x * (1 - x)
                # Map 0-1 to lb-ub
                chaotic_pos[i, d] = self.lb + x * (self.ub - self.lb)
        return chaotic_pos

    def opposition_based_learning(self):
        """
        IMPROVEMENT 2: Opposition-Based Learning (OBL)
        Check the 'opposite' of current positions. If better, swap.
        """
        # Select a random subset of sparrows to apply OBL (e.g., 10%)
        candidates_idx = np.random.choice(self.num_sparrows, max(1, int(0.1 * self.num_sparrows)), replace=False)
        
        for i in candidates_idx:
            current_x = self.positions[i]
            current_f = self.fitness[i]
            
            # Calculate opposite: a + b - x
            opposite_x = self.lb + self.ub - current_x
            
            # Ensure bounds
            opposite_x = np.clip(opposite_x, self.lb, self.ub)
            
            # Evaluate
            opposite_f = self.objective_function(opposite_x)
            
            # Greedy selection
            if opposite_f < current_f:
                self.positions[i] = opposite_x
                self.fitness[i] = opposite_f

    def calculate_potential_forces(self, current_idx):
        """
        IMPROVEMENT 4: Potential Field / Swarm Repulsion
        Calculates repulsive forces from nearby sparrows to maintain diversity.
        This prevents the swarm from collapsing into a local optimum too early.
        """
        current_pos = self.positions[current_idx]
        force = np.zeros(self.dimensions)
        
        # Repulsive force from other sparrows
        for i in range(self.num_sparrows):
            if i != current_idx:
                diff = current_pos - self.positions[i]
                dist = np.linalg.norm(diff)
                
                # Only repel if very close (radius of influence)
                # Heuristic: 5% of the search space size
                repulsion_radius = (self.ub - self.lb) * 0.05
                
                if dist < repulsion_radius and dist > 1e-10:
                    # Inverse square law repulsion
                    f_mag = 0.5 / (dist**2)
                    force += f_mag * (diff / dist)
        
        # Cap the force to prevent explosions
        max_force = (self.ub - self.lb) * 0.05 # Max kick is 5% of space
        force_mag = np.linalg.norm(force)
        if force_mag > max_force:
            force = force / force_mag * max_force
            
        return force

    def update_roles(self):
        sorted_indices = np.argsort(self.fitness)
        # Dynamic producers count
        num_producers = max(1, int(0.2 * self.num_sparrows + np.random.randint(-2, 3)))
        self.producers = sorted_indices[:num_producers]
        self.scroungers = sorted_indices[num_producers:]
        
        self.producers_count.append(len(self.producers))
        self.scroungers_count.append(len(self.scroungers))
    
    def update_producers(self, iter_num):
        R2 = np.random.rand()
        ST = 0.6
        
        # IMPROVEMENT 3: Adaptive Weighting
        # w decreases linearly from 0.9 to 0.4.
        # High inertia early (exploration), low inertia late (exploitation).
        w = 0.9 - 0.5 * (iter_num / self.max_iter)
        
        for i in self.producers:
            # Calculate Repulsive Force (Physics Layer)
            repulsion = self.calculate_potential_forces(i)
            
            if R2 < ST:
                # Classic SSA producer move + Adaptive Weight + Repulsion
                self.positions[i] = self.positions[i] * np.exp(-i / (0.8 * self.max_iter))
                # Add physics nudge
                self.positions[i] += repulsion
            else:
                step = np.random.randn(self.dimensions)
                # Adaptive exploration
                self.positions[i] += step * w 
                
            self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
    
    def update_scroungers(self, iter_num):
        # Adaptive weight
        w = 0.9 - 0.5 * (iter_num / self.max_iter)
        
        for i in self.scroungers:
            # Calculate Repulsive Force (Physics Layer)
            repulsion = self.calculate_potential_forces(i)
            
            if self.fitness[i] > np.median(self.fitness):
                # Move towards best
                # Standard: self.positions[i] += np.random.rand() * (self.best_pos - self.positions[i])
                # Enhanced: Add Weight and Repulsion
                self.positions[i] = w * self.positions[i] + \
                                    np.random.rand() * (self.best_pos - self.positions[i]) + \
                                    repulsion
            else:
                # Local exploration
                self.positions[i] += np.random.uniform(-0.1, 0.1, self.dimensions) * w
                
            self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
            
            # Competition
            new_fitness = self.objective_function(self.positions[i])
            if new_fitness < self.fitness[self.producers[-1]]:
                self.positions[self.producers[-1]] = self.positions[i].copy()
                self.fitness[self.producers[-1]] = new_fitness
    
    def anti_predator_escape(self):
        worst_index = np.argmax(self.fitness)
        worst_pos = self.positions[worst_index]
        for i in range(self.num_sparrows):
            escape_factor = np.random.uniform(-1, 1, self.dimensions)
            self.positions[i] += escape_factor * (self.positions[i] - worst_pos) * 0.1
            self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
    
    def random_walk(self, t, max_t):
        """Levy-like random walk logic"""
        r_t = np.random.rand(self.dimensions)
        r_t = np.where(r_t > 0.5, 1, 0)
        steps = 2 * r_t - 1
        X_t = np.cumsum(steps)
        
        progress_ratio = t / max_t
        a = np.min(self.positions, axis=0)
        b = np.max(self.positions, axis=0)
        
        c = self.best_pos - (1 - progress_ratio) * (b - a) * 0.5
        d = self.best_pos + (1 - progress_ratio) * (b - a) * 0.5
        
        X_normalized = (X_t - np.min(X_t)) * (d - c) / (np.max(X_t) - np.min(X_t) + 1e-10) + c
        X_normalized = np.clip(X_normalized, self.lb, self.ub)
        return X_normalized

    def optimize(self):
        for iter_num in range(self.max_iter):
            self.update_roles()
            self.history.append((self.positions.copy(), self.best_pos.copy(), iter_num+1))
            
            # Update Producers (includes Adaptive Weights & Potential Fields)
            self.update_producers(iter_num)
            
            # Update Scroungers (includes Adaptive Weights & Potential Fields)
            self.update_scroungers(iter_num)
            
            # Random Walk Strategy
            if np.random.rand() < 0.3:
                random_walk_pos = self.random_walk(iter_num, self.max_iter)
                random_walk_fitness = self.objective_function(random_walk_pos)
                if random_walk_fitness < self.best_fit:
                    self.best_pos = random_walk_pos
                    self.best_fit = random_walk_fitness
                    worst_idx = np.argmax(self.fitness)
                    self.positions[worst_idx] = random_walk_pos
                    self.fitness[worst_idx] = random_walk_fitness
            
            self.anti_predator_escape()
            
            # IMPROVEMENT 2: Opposition-Based Learning
            # Run periodically (e.g., every 5 iterations) to save computation
            if iter_num % 5 == 0:
                self.opposition_based_learning()
            
            # Update Global Best
            self.fitness = np.apply_along_axis(self.objective_function, 1, self.positions)
            best_index = np.argmin(self.fitness)
            if self.fitness[best_index] < self.best_fit:
                self.best_fit = self.fitness[best_index]
                self.best_pos = self.positions[best_index].copy()
            
            self.convergence_history.append(self.best_fit)
        
        return self.best_pos, self.best_fit
    
    def visualize(self):
        # Only visualize if 2D or 3D (projected to 2D)
        if self.dimensions > 2:
            print("Visualization is optimized for 2D. Showing first 2 dimensions.")
            
        x = np.linspace(self.lb, self.ub, 100)
        y = np.linspace(self.lb, self.ub, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = np.zeros(self.dimensions)
                point[0] = X[i, j]
                point[1] = Y[i, j]
                Z[i, j] = self.objective_function(point)

        plt.figure(figsize=(12, 8))
        
        # Use a subset of history for speed if too long
        display_steps = self.history
        if len(self.history) > 50:
            display_steps = self.history[::2] # Skip frames for speed

        for positions, best, iter_num in display_steps:
            plt.clf()
            
            gs = GridSpec(2, 2, height_ratios=[3, 1])
            
            # Main plot
            ax1 = plt.subplot(gs[0, :])
            contour = ax1.contourf(X, Y, Z, levels=50, cmap='viridis')
            ax1.scatter(positions[:, 0], positions[:, 1], c='gray', alpha=0.5, label='Sparrows')
            
            # Recalculate roles for visualization color (approximate based on sorting)
            # Since we didn't store role indices per history step, we re-infer for viz
            # This is a visual approximation
            fitness_approx = np.apply_along_axis(self.objective_function, 1, positions)
            sorted_idx = np.argsort(fitness_approx)
            num_prod = self.producers_count[min(iter_num-1, len(self.producers_count)-1)]
            producers_idx = sorted_idx[:num_prod]
            scroungers_idx = sorted_idx[num_prod:]
            
            ax1.scatter(positions[producers_idx, 0], positions[producers_idx, 1], c='orange', label='Producers')
            ax1.scatter(positions[scroungers_idx, 0], positions[scroungers_idx, 1], c='blue', label='Scroungers')
            ax1.scatter(best[0], best[1], c='red', marker='x', s=100, label='Global Best')
            
            ax1.set_xlim(self.lb, self.ub)
            ax1.set_ylim(self.lb, self.ub)
            ax1.set_title(f"Enhanced COB-SSA - Iteration {iter_num}")
            ax1.legend(loc='upper right')
            
            # Role Counts
            ax2 = plt.subplot(gs[1, 0])
            iterations = list(range(1, len(self.producers_count) + 1))
            ax2.plot(iterations, self.producers_count, 'orange', label='Producers')
            ax2.plot(iterations, self.scroungers_count, 'blue', label='Scroungers')
            ax2.set_xlim(1, self.max_iter)
            ax2.set_ylim(0, self.num_sparrows)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Count')
            ax2.legend(loc='upper right')
            ax2.grid(True)
            if iter_num <= len(self.producers_count):
                ax2.axvline(x=iter_num, color='r', linestyle='--')
            
            # Convergence
            ax3 = plt.subplot(gs[1, 1])
            if len(self.convergence_history) > 0:
                conv_iters = list(range(1, len(self.convergence_history) + 1))
                ax3.plot(conv_iters, self.convergence_history, 'g-', label='Best Fitness')
                ax3.set_xlim(1, self.max_iter)
                ax3.set_yscale('log') # Log scale is better for convergence
                ax3.set_xlabel('Iteration')
                ax3.set_ylabel('Fitness (Log)')
                ax3.set_title('Convergence')
                ax3.grid(True)
                if iter_num <= len(self.convergence_history):
                    ax3.axvline(x=iter_num, color='r', linestyle='--')
            
            plt.tight_layout()
            plt.pause(0.01) # Faster animation
        
        plt.show()

    def plot_final_results(self):
        """Plot the final convergence curve and producer/scrounger counts"""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        iterations = list(range(1, len(self.convergence_history) + 1))
        plt.plot(iterations, self.convergence_history, 'g-')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness (Log Scale)')
        plt.yscale('log')
        plt.title('Final Convergence Curve')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(iterations, self.producers_count, 'orange', label='Producers')
        plt.plot(iterations, self.scroungers_count, 'blue', label='Scroungers')
        plt.xlabel('Iteration')
        plt.ylabel('Count')
        plt.title('Dynamic Role Assignment')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

def compare_convergence(results):
    plt.figure(figsize=(10, 6))
    for name, convergence in results.items():
        plt.plot(range(1, len(convergence) + 1), convergence, label=name)
    
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness (log scale)')
    plt.title('Convergence Comparison of Test Functions')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

def run_all_functions(visualize=True):
    functions = {
        "Sphere Function": sphere_function,
        "Rosenbrock Function": rosenbrock_function,
        "Ackley Function": ackley_function,
        "Weighted Quadratic": weighted_quadratic,
        "Beale Variant": beale_variant,
        "Schwefel Function": schwefel_function
    }
    
    results = {}
    
    print("Running Enhanced COB-SSA (Chaotic Opposition-Based SSA with Potential Fields)...")
    
    for name, func in functions.items():
        print(f"\n--- {name} ---")
        # Initialize with typical parameters
        ssa = EnhancedSparrowSearchAlgorithm(
            objective_function=func,
            num_sparrows=50, 
            dimensions=2, # Keep to 2 for good visualization, change to 30+ for real testing
            max_iter=100,
            lb=-5, # Some functions like Schwefel need wider bounds, but -5 to 5 is standard for most
            ub=5
        )
        
        # Adjust bounds for Schwefel specifically as it needs large range
        if name == "Schwefel Function":
            ssa.lb, ssa.ub = -500, 500
            ssa.positions = ssa.chaotic_initialization() # Re-init with new bounds
            
        best_solution, best_value = ssa.optimize()
        print(f"Best Solution: {best_solution}")
        print(f"Best Objective Value: {best_value:.6e}")
        
        results[name] = ssa.convergence_history
        
        if visualize:
            ssa.visualize()
        else:
            ssa.plot_final_results()
    
    compare_convergence(results)

if __name__ == "__main__":
    # Set visualize=False to just run fast and see final plots
    # Set visualize=True to see the swarm animation
    run_all_functions(visualize=True)