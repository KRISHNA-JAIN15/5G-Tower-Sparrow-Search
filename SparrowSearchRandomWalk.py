import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def sphere_function(x):
    return np.sum(x ** 2)

def rosenbrock_function(x):
    return sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

def ackley_function(x):
    a, b, c = 20, 0.2, 2 * np.pi
    d = len(x)
    return -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / d)) - np.exp(np.sum(np.cos(c * x)) / d) + a + np.exp(1)

def weighted_quadratic(x):
    # Add varying weights to each dimension
    a = np.linspace(1, 10, len(x))
    return np.sum(a * x**2)

def beale_variant(x):
    x0, x1 = x[0], x[1]
    term1 = (1.5 - x0 + x0 * x1)**2
    term2 = (2.25 - x0 + x0 * x1**2)**2
    return term1 + term2

def schwefel_function(x):
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

class SparrowSearchAlgorithm:
    def __init__(self, objective_function, num_sparrows=30, dimensions=3, max_iter=100, lb=-3, ub=3):
        self.objective_function = objective_function
        self.num_sparrows = num_sparrows
        self.dimensions = dimensions
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        self.positions = np.random.uniform(lb - 0.1, ub, (num_sparrows, dimensions))  # Start near the upper bound
        self.fitness = np.apply_along_axis(self.objective_function, 1, self.positions)
        self.best_pos = self.positions[np.argmin(self.fitness)]
        self.best_fit = np.min(self.fitness)
        self.history = []
        self.producers_count = []  # Track number of producers in each iteration
        self.scroungers_count = []  # Track number of scroungers in each iteration
        self.convergence_history = []  # Track best fitness in each iteration
        # Initialize producers and scroungers for first visualization
        self.producers = []
        self.scroungers = []
    
    def update_roles(self):
        sorted_indices = np.argsort(self.fitness)
        num_producers = max(1, int(0.2 * self.num_sparrows + np.random.randint(-2, 3)))  # Dynamic producers
        self.producers = sorted_indices[:num_producers]
        self.scroungers = sorted_indices[num_producers:]
        
        # Store counts for visualization
        self.producers_count.append(len(self.producers))
        self.scroungers_count.append(len(self.scroungers))
    
    def update_producers(self):
        R2 = np.random.rand()
        ST = 0.6  # Safety threshold
        for i in self.producers:
            step = np.random.randn(self.dimensions)
            if R2 < ST:
                self.positions[i] += step * 0.1
            else:
                self.positions[i] += np.random.uniform(-1, 1, self.dimensions) * 0.5  # Escape move
            self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
    
    def update_scroungers(self):
        for i in self.scroungers:
            if self.fitness[i] > np.median(self.fitness):  # "Hungry" joiners move towards best
                self.positions[i] += np.random.rand() * (self.best_pos - self.positions[i])
            else:  # "Well-fed" joiners explore locally
                self.positions[i] += np.random.uniform(-0.1, 0.1, self.dimensions)
            self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
            
            # Competition between joiners and producers
            new_fitness = self.objective_function(self.positions[i])
            if new_fitness < self.fitness[self.producers[-1]]:  # If a joiner performs better than the worst producer
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
        """
        Implement random walk to perturb the optimal sparrow position.
        
        Parameters:
        t (int): Current iteration number
        max_t (int): Maximum number of iterations
        
        Returns:
        np.ndarray: Random walk step
        """
        # Generate random binary values according to equation (11)
        r_t = np.random.rand(self.dimensions)
        r_t = np.where(r_t > 0.5, 1, 0)
        
        # Calculate cumulative sum as in equation (10)
        steps = 2 * r_t - 1
        X_t = np.cumsum(steps)
        
        # Calculate dynamic boundaries for random walk
        # As iterations progress, search range decreases (improve local search)
        progress_ratio = t / max_t
        a = np.min(self.positions, axis=0)
        b = np.max(self.positions, axis=0)
        
        # Dynamic values for c and d to control random walk boundaries
        c = self.best_pos - (1 - progress_ratio) * (b - a) * 0.5
        d = self.best_pos + (1 - progress_ratio) * (b - a) * 0.5
        
        # Normalize according to equation (12)
        X_normalized = (X_t - np.min(X_t)) * (d - c) / (np.max(X_t) - np.min(X_t) + 1e-10) + c
        
        # Ensure the walk stays within bounds
        X_normalized = np.clip(X_normalized, self.lb, self.ub)
        
        return X_normalized

    def optimize(self):
        for iter_num in range(self.max_iter):
            self.update_roles()
            self.history.append((self.positions.copy(), self.best_pos.copy(), iter_num+1))
            self.update_producers()
            self.update_scroungers()
            
            # Apply random walk to the best position with probability
            if np.random.rand() < 0.3:  # 30% chance to perform random walk
                random_walk_pos = self.random_walk(iter_num, self.max_iter)
                random_walk_fitness = self.objective_function(random_walk_pos)
                
                # If random walk position is better, update best position
                if random_walk_fitness < self.best_fit:
                    self.best_pos = random_walk_pos
                    self.best_fit = random_walk_fitness
                    
                    # Replace worst sparrow with this new position
                    worst_idx = np.argmax(self.fitness)
                    self.positions[worst_idx] = random_walk_pos
                    self.fitness[worst_idx] = random_walk_fitness
            
            self.anti_predator_escape()
            
            self.fitness = np.apply_along_axis(self.objective_function, 1, self.positions)
            best_index = np.argmin(self.fitness)
            if self.fitness[best_index] < self.best_fit:
                self.best_fit = self.fitness[best_index]
                self.best_pos = self.positions[best_index]
            
            # Store best fitness for convergence plot
            self.convergence_history.append(self.best_fit)
        
        return self.best_pos, self.best_fit
    
    def visualize(self):
        x = np.linspace(self.lb, self.ub, 100)
        y = np.linspace(self.lb, self.ub, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        # Use 2D input for the objective function visualization
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                # Create a point with dimensions matching our problem
                point = np.zeros(self.dimensions)
                point[0] = X[i, j]
                point[1] = Y[i, j]
                Z[i, j] = self.objective_function(point)

        plt.figure(figsize=(12, 8))
        
        for positions, best, iter_num in self.history:
            plt.clf()
            
            # Create a 2x2 grid for subplots
            gs = GridSpec(2, 2, height_ratios=[3, 1])
            
            # Main plot for search visualization
            ax1 = plt.subplot(gs[0, :])
            contour = ax1.contourf(X, Y, Z, levels=50, cmap='viridis')
            
            # Plot only the first two dimensions
            ax1.scatter(positions[:, 0], positions[:, 1], c='gray', alpha=0.5, label='Sparrows')
            ax1.scatter(positions[self.producers, 0], positions[self.producers, 1], c='orange', label='Producers')
            ax1.scatter(positions[self.scroungers, 0], positions[self.scroungers, 1], c='blue', label='Scroungers')
            ax1.scatter(best[0], best[1], c='red', marker='x', s=100, label='Best Solution')
            ax1.set_xlim(self.lb, self.ub)
            ax1.set_ylim(self.lb, self.ub)
            ax1.set_title(f"Iteration {iter_num}")
            ax1.legend(loc='upper right')
            
            # Second plot for role counts
            ax2 = plt.subplot(gs[1, 0])
            iterations = list(range(1, len(self.producers_count) + 1))
            ax2.plot(iterations, self.producers_count, 'orange', label='Producers')
            ax2.plot(iterations, self.scroungers_count, 'blue', label='Scroungers')
            ax2.set_xlim(1, self.max_iter)
            ax2.set_ylim(0, self.num_sparrows)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Count')
            ax2.set_title('Number of Producers and Scroungers')
            ax2.legend(loc='upper right')
            ax2.grid(True)
            
            # Highlight current iteration
            if iter_num <= len(self.producers_count):
                ax2.axvline(x=iter_num, color='r', linestyle='--')
                
                # Add text displaying current counts
                prod_count = self.producers_count[iter_num-1]
                scr_count = self.scroungers_count[iter_num-1]
                ax2.text(0.02, 0.9, f"Current: {prod_count} Producers, {scr_count} Scroungers", 
                         transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.8))
            
            # Third plot for convergence
            ax3 = plt.subplot(gs[1, 1])
            if len(self.convergence_history) > 0:
                convergence_iterations = list(range(1, len(self.convergence_history) + 1))
                ax3.plot(convergence_iterations, self.convergence_history, 'g-', label='Best Fitness')
                ax3.set_xlim(1, self.max_iter)
                # Set y-axis limits based on current data with some margin
                if len(self.convergence_history) > 0:
                    min_val = min(self.convergence_history)
                    max_val = max(self.convergence_history[:iter_num]) if iter_num < len(self.convergence_history) else max(self.convergence_history)
                    margin = (max_val - min_val) * 0.1 if max_val > min_val else 0.1
                    ax3.set_ylim(max(0, min_val - margin), max_val + margin)
                ax3.set_xlabel('Iteration')
                ax3.set_ylabel('Best Fitness')
                ax3.set_title('Convergence Curve')
                ax3.grid(True)
                
                # Highlight current iteration
                if iter_num <= len(self.convergence_history):
                    ax3.axvline(x=iter_num, color='r', linestyle='--')
                    current_best = self.convergence_history[iter_num-1]
                    ax3.scatter(iter_num, current_best, c='r', s=50)
                    ax3.text(0.02, 0.9, f"Current Best: {current_best:.6f}", 
                             transform=ax3.transAxes, bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.pause(0.1)
        
        plt.show()

    def plot_final_results(self):
        """Plot the final convergence curve and producer/scrounger counts"""
        plt.figure(figsize=(12, 6))
        
        # Plot convergence
        plt.subplot(1, 2, 1)
        iterations = list(range(1, len(self.convergence_history) + 1))
        plt.plot(iterations, self.convergence_history, 'g-')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness')
        plt.title('Convergence Curve')
        plt.grid(True)
        
        # Plot producer/scrounger counts
        plt.subplot(1, 2, 2)
        plt.plot(iterations, self.producers_count, 'orange', label='Producers')
        plt.plot(iterations, self.scroungers_count, 'blue', label='Scroungers')
        plt.xlabel('Iteration')
        plt.ylabel('Count')
        plt.title('Number of Producers and Scroungers')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

def compare_convergence(results):
    """Compare convergence across different functions"""
    plt.figure(figsize=(10, 6))
    for name, convergence in results.items():
        plt.plot(range(1, len(convergence) + 1), convergence, label=name)
    
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness (log scale)')
    plt.title('Convergence Comparison')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Using log scale for better visualization
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
    
    for name, func in functions.items():
        print(f"Running SSA on {name}...")
        ssa = SparrowSearchAlgorithm(objective_function=func)
        best_solution, best_value = ssa.optimize()
        print("Best Solution:", best_solution)
        print("Best Objective Value:", best_value)
        
        # Store convergence history for comparison
        results[name] = ssa.convergence_history
        
        # Show animation if requested
        if visualize:
            ssa.visualize()
        
        # Always show the final results
        ssa.plot_final_results()
    
    # Compare convergence across all functions
    compare_convergence(results)

# Run the algorithm
if __name__ == "__main__":
    run_all_functions(visualize=True)