import numpy as np
import matplotlib.pyplot as plt

def perception_probability(distance, Rs, Re=1.0, lambda_param=1.0):
    """
    Three-part perception probability model:
    p(zi,Hj) = 0, if d(zi,Hj) >= Rs
              = exp(-λ(Rs-d(zi,Hj))/(d(zi,Hj)-Rs-Re)), if Rs-Re < d(zi,Hj) < Rs
              = 1, if d(zi,Hj) <= Rs-Re
    """
    if np.isscalar(distance):
        if distance >= Rs:
            return 0
        elif distance <= Rs - Re:
            return 1
        else:
            return np.exp(-lambda_param * (Rs - distance) / (distance - Rs - Re))
    else:  # For array processing
        result = np.zeros_like(distance, dtype=float)
        mask_far = distance >= Rs
        mask_close = distance <= Rs - Re
        mask_middle = ~mask_far & ~mask_close
        
        result[mask_far] = 0
        result[mask_close] = 1
        result[mask_middle] = np.exp(-lambda_param * (Rs - distance[mask_middle]) / (distance[mask_middle] - Rs - Re))
        return result

def joint_probability(base_stations, X, Y, Rs, Re=1.0, lambda_param=1.0):
    """
    Calculate joint perception probability using the product formula:
    p(Z,Hj) = 1 - ∏[1 - p(zi,Hj)]
    """
    joint_prob = np.ones_like(X, dtype=float)
    
    for x, y in base_stations:
        distances = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
        prob = perception_probability(distances, Rs, Re, lambda_param)
        joint_prob *= (1 - prob)
    
    # Final joint probability
    return 1 - joint_prob

def coverage_ratio(base_stations, area_size, grid_size, Rs, Re=1.0, lambda_param=1.0):
    """
    Calculate coverage ratio using joint probability
    Rcov = ∑p(Z,Hj) / (m×n)
    """
    L1, L2 = area_size
    m, n = grid_size
    x_grid = np.linspace(0, L1, m)
    y_grid = np.linspace(0, L2, n)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Calculate joint probability for each grid point
    prob_grid = joint_probability(base_stations, X, Y, Rs, Re, lambda_param)
    
    # Calculate coverage ratio
    total_coverage = np.sum(prob_grid)
    total_points = m * n
    
    print(f"Total Area Coverage: {total_coverage / total_points * 100:.2f}%")
    return total_coverage / total_points

def random_walk_normalization(X, a, b, c, d):
    """
    Implements the random walk normalization as described in equation 12:
    X^t = ((X^t - a_i) / (d_i - c_i)) * (b_i - a_i) + c_i
    
    Parameters:
    - X: Current position
    - a, b: Lower and upper bounds of the search space
    - c, d: Minimum and maximum values of the random walk
    """
    return (b - a) * ((X - a) / (d - c)) + c

class RandomWalkSparrowSearchAlgorithm:
    def __init__(self, num_sparrows=30, num_stations=5, area_size=(10, 10), grid_size=(50, 50), 
                 Rs=2.2, Re=1.0, lambda_param=1.0, max_iter=100):
        self.num_sparrows = num_sparrows
        self.num_stations = num_stations
        self.area_size = area_size
        self.grid_size = grid_size
        self.Rs = Rs
        self.Re = Re
        self.lambda_param = lambda_param
        self.max_iter = max_iter
        self.positions = self.initialize_positions()
        self.fitness = np.array([coverage_ratio(pos, area_size, grid_size, Rs, Re, lambda_param) for pos in self.positions])
        self.best_pos = self.positions[np.argmax(self.fitness)]
        self.best_fit = np.max(self.fitness)
        self.history = []
        
        # Parameters for random walk
        self.L1, self.L2 = area_size
        self.a = 0.0  # Lower bound
        self.b = self.L1  # Upper bound
    
    def initialize_positions(self):
        valid_positions = np.zeros((self.num_sparrows, self.num_stations, 2))
        min_dist = 2.0  # Minimum distance between stations
        min_boundary_dist = 2.0  # Minimum distance from boundaries
        L1, L2 = self.area_size
        
        for i in range(self.num_sparrows):
            stations = []
            while len(stations) < self.num_stations:
                x, y = np.random.uniform(min_boundary_dist, L1 - min_boundary_dist), np.random.uniform(min_boundary_dist, L2 - min_boundary_dist)
                if all(np.linalg.norm(np.array([x, y]) - np.array(s)) >= min_dist for s in stations):
                    stations.append((x, y))
            valid_positions[i] = np.array(stations)
        return valid_positions
    
    def update_roles(self):
        sorted_indices = np.argsort(-self.fitness)
        num_producers = max(1, int(0.2 * self.num_sparrows))
        self.producers = sorted_indices[:num_producers]
        self.scroungers = sorted_indices[num_producers:]
    
    def enforce_constraints(self, positions):
        """Ensure base stations stay within bounds and maintain minimum distance"""
        min_dist = 2.0
        min_boundary_dist = 2.0
        L1, L2 = self.area_size
        
        for i in range(self.num_stations):
            positions[i] = np.clip(positions[i], min_boundary_dist, L1 - min_boundary_dist)
            for j in range(i):
                while np.linalg.norm(positions[i] - positions[j]) < min_dist:
                    positions[i] = np.random.uniform(min_boundary_dist, L1 - min_boundary_dist, 2)
        return positions
    
    def update_producers(self):
        """
        Update discoverers (producers) position following equation 7:
        Xi,j^(t+1) = {
            Xi,j*exp(-i/(α*iter_max)), if R2 < ST
            Xi,j + Q*L, if R2 >= ST
        }
        """
        alpha = 0.8  # Constant for equation
        ST = 0.6     # Safety threshold
        Q = 0.1      # Step size
        R2 = np.random.rand()  # Random value for condition
        
        for i in self.producers:
            curr_iter = len(self.history) + 1
            if R2 < ST:
                # Cautious movement
                exp_factor = np.exp(-i / (alpha * self.max_iter))
                self.positions[i] = self.positions[i] * exp_factor
            else:
                # Explore new areas
                L = np.random.normal(0, 1, self.positions[i].shape)  # Random direction
                self.positions[i] = self.positions[i] + Q * L
                
            # Apply random walk for some producers (20% chance)
            if np.random.rand() < 0.2:
                self.apply_random_walk(i)
                
            # Enforce constraints
            self.positions[i] = self.enforce_constraints(self.positions[i])
    
    def apply_random_walk(self, index):
        """
        Implement random walk as described in equations 10-11:
        X(t) = [0, cumsum(2r(t1)-1), ..., cumsum(2r(tn)-1)]
        where r(t) = 1 if rand > 0.5, 0 otherwise
        """
        # Generate random binary sequence
        steps = 100
        r = np.random.rand(steps) > 0.5
        r = 2 * r.astype(int) - 1  # Convert to -1 or 1
        
        # Calculate cumulative sum for the walk
        walk_x = np.zeros(steps + 1)
        walk_y = np.zeros(steps + 1)
        walk_x[1:] = np.cumsum(r)
        walk_y[1:] = np.cumsum(np.random.rand(steps) > 0.5) * 2 - 1
        
        # Normalize to current position ranges
        min_x, max_x = np.min(walk_x), np.max(walk_x)
        min_y, max_y = np.min(walk_y), np.max(walk_y)
        
        for j in range(self.num_stations):
            # Get a random point from the walk
            idx = np.random.randint(0, steps)
            
            # Apply normalized position using equation 12
            pos_x = self.positions[index][j][0]
            pos_y = self.positions[index][j][1]
            
            # Normalize and apply the walk
            new_x = random_walk_normalization(walk_x[idx], min_x, max_x, 0, self.L1)
            new_y = random_walk_normalization(walk_y[idx], min_y, max_y, 0, self.L2)
            
            # Combine current position with random walk influence
            self.positions[index][j][0] = 0.7 * pos_x + 0.3 * new_x
            self.positions[index][j][1] = 0.7 * pos_y + 0.3 * new_y
    
    def update_scroungers(self):
        """
        Update joiners (scroungers) position following equations 8-9:
        Uses simplified versions appropriate for base station placement
        """
        best_producer_idx = self.producers[0]  # Get best producer
        
        for i in self.scroungers:
            if np.random.rand() > 0.8:  # 20% chance for alert behavior
                # Move away from worst position (anti-predator)
                worst_idx = np.argmin(self.fitness)
                K = np.random.uniform(-1, 1)
                self.positions[i] += K * np.abs(self.positions[i] - self.positions[worst_idx])
            else:
                # Normal joiner behavior - follow best producer
                A = np.random.randn(self.positions[i].shape[0], self.positions[i].shape[1])
                self.positions[i] += np.random.rand() * A * (self.positions[best_producer_idx] - self.positions[i])
            
            # Enforce constraints
            self.positions[i] = self.enforce_constraints(self.positions[i])
    
    def anti_predator_escape(self):
        """
        Implement anti-predator behavior for the lowest-performing sparrows
        """
        worst_idx = np.argmin(self.fitness)
        # Select 10% worst performers to apply escape
        num_escape = max(1, int(0.1 * self.num_sparrows))
        escape_indices = np.argsort(self.fitness)[:num_escape]
        
        for i in escape_indices:
            if np.random.rand() < 0.8:  # 80% chance to escape
                # Random jump to a new location
                self.positions[i] = self.initialize_positions()[0]
            else:
                # Move towards global best
                self.positions[i] += np.random.normal(0, 0.5, self.positions[i].shape) * (self.best_pos - self.positions[i])
            
            # Enforce constraints
            self.positions[i] = self.enforce_constraints(self.positions[i])
    
    def optimize(self):
        """Run the optimization process"""
        for iter in range(self.max_iter):
            print(f"Iteration {iter+1}/{self.max_iter}")
            self.history.append((self.positions.copy(), self.best_pos.copy(), self.best_fit))
            
            # Update roles based on fitness
            self.update_roles()
            
            # Update positions
            self.update_producers()
            self.update_scroungers()
            
            # Apply anti-predator behavior
            if iter % 5 == 0:  # Every 5 iterations
                self.anti_predator_escape()
            
            # Calculate new fitness values
            self.fitness = np.array([coverage_ratio(pos, self.area_size, self.grid_size, 
                                                   self.Rs, self.Re, self.lambda_param) 
                                    for pos in self.positions])
            
            # Update best solution if improved
            best_index = np.argmax(self.fitness)
            if self.fitness[best_index] > self.best_fit:
                self.best_fit = self.fitness[best_index]
                self.best_pos = self.positions[best_index].copy()
                print(f"New best solution found! Coverage: {self.best_fit * 100:.2f}%")
            
            # Print current best
            print(f"Current best coverage: {self.best_fit * 100:.2f}%")
            
        return self.best_pos, self.best_fit
    
    def visualize_coverage(self):
        """Visualize the coverage of the best solution"""
        L1, L2 = self.area_size
        m, n = self.grid_size
        x_grid = np.linspace(0, L1, m)
        y_grid = np.linspace(0, L2, n)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Calculate probability grid using joint probability
        prob_grid = joint_probability(self.best_pos, X, Y, self.Rs, self.Re, self.lambda_param)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        # Plot joint probability as a heatmap
        contour = plt.contourf(X, Y, prob_grid, levels=20, cmap='viridis', alpha=0.7)
        plt.colorbar(contour, label='Coverage Probability')
        
        # Plot base stations
        plt.scatter(self.best_pos[:, 0], self.best_pos[:, 1], color='red', marker='x', s=100, label='Base Stations')
        
        # Plot coverage circles
        for x, y in self.best_pos:
            circle_outer = plt.Circle((x, y), self.Rs, fill=False, color='red', linestyle='-', alpha=0.5)
            circle_inner = plt.Circle((x, y), self.Rs - self.Re, fill=False, color='green', linestyle='--', alpha=0.5)
            plt.gca().add_patch(circle_outer)
            plt.gca().add_patch(circle_inner)
        
        plt.title(f"Base Station Coverage (Max Coverage: {self.best_fit * 100:.2f}%)")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def visualize_convergence(self):
        """Plot the convergence history"""
        iterations = range(1, len(self.history) + 1)
        best_fits = [h[2] for h in self.history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, [fit * 100 for fit in best_fits], 'b-', marker='o', markersize=4)
        plt.title("Convergence History")
        plt.xlabel("Iteration")
        plt.ylabel("Best Coverage Ratio (%)")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

# Main execution
if __name__ == "__main__":
    print("Running Random Walk SSA for Base Station Placement Optimization...")
    
    # Parameters
    area_size = (15, 15)  # Area size (L1, L2)
    grid_size = (100,100)
    Rs = 2.2  # Coverage radius
    Re = 1.0  # Inner radius for probability model
    lambda_param = 1.0  # Parameter for probability exponential decay
    
    # Create and run the optimization
    rwssa = RandomWalkSparrowSearchAlgorithm(
        num_sparrows=30,
        num_stations=5,
        area_size=area_size,
        grid_size=grid_size,
        Rs=Rs,
        Re=Re,
        lambda_param=lambda_param,
        max_iter=100
    )
    
    # Run optimization
    best_solution, best_value = rwssa.optimize()
    
    # Print results
    print("\nOptimization Complete!")
    print("Best Base Station Positions:\n", best_solution)
    print(f"Maximum Coverage Ratio: {best_value * 100:.2f}%")
    
    # Visualize results
    rwssa.visualize_coverage()
    rwssa.visualize_convergence()