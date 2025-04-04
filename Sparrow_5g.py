# import numpy as np
# import matplotlib.pyplot as plt

# def is_covered(distance, Rs):
#     return distance <= Rs

# def signal_strength(distance, Rs):
#     return np.exp(-distance / Rs)

# def coverage_ratio(base_stations, area_size, grid_size, Rs):
#     L1, L2 = area_size
#     m, n = grid_size
#     x_grid = np.linspace(0, L1, m)
#     y_grid = np.linspace(0, L2, n)
#     X, Y = np.meshgrid(x_grid, y_grid)
#     coverage_grid = np.zeros_like(X, dtype=int)
    
#     for x, y in base_stations:
#         distances = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
#         coverage_grid |= is_covered(distances, Rs)
    
#     total_covered = np.sum(coverage_grid)
#     total_points = m * n
#     print(f"Total Area Coverage: {total_covered / total_points * 100:.2f}%")
    
#     return total_covered / total_points

# class SparrowSearchAlgorithm:
#     def __init__(self, num_sparrows=30, num_stations=5, area_size=(10, 10), grid_size=(50, 50), Rs=2.2, max_iter=100):
#         self.num_sparrows = num_sparrows
#         self.num_stations = num_stations
#         self.area_size = area_size
#         self.grid_size = grid_size
#         self.Rs = Rs
#         self.max_iter = max_iter
#         self.positions = self.initialize_positions()
#         self.fitness = np.array([coverage_ratio(pos, area_size, grid_size, Rs) for pos in self.positions])
#         self.best_pos = self.positions[np.argmax(self.fitness)]
#         self.best_fit = np.max(self.fitness)
#         self.history = []
    
#     def initialize_positions(self):
#         valid_positions = np.zeros((self.num_sparrows, self.num_stations, 2))
#         min_dist = 3.1
#         min_boundary_dist = 1
#         L1, L2 = self.area_size
        
#         for i in range(self.num_sparrows):
#             stations = []
#             while len(stations) < self.num_stations:
#                 x, y = np.random.uniform(min_boundary_dist, L1 - min_boundary_dist), np.random.uniform(min_boundary_dist, L2 - min_boundary_dist)
#                 if all(np.linalg.norm(np.array([x, y]) - np.array(s)) >= min_dist for s in stations):
#                     stations.append((x, y))
#             valid_positions[i] = np.array(stations)
#         return valid_positions
    
#     def update_roles(self):
#         sorted_indices = np.argsort(-self.fitness)
#         num_producers = max(1, int(0.2 * self.num_sparrows))
#         self.producers = sorted_indices[:num_producers]
#         self.scroungers = sorted_indices[num_producers:]
    
#     def enforce_constraints(self, positions):
#         min_dist = 3.1
#         min_boundary_dist = 1
#         L1, L2 = self.area_size
        
#         for i in range(self.num_stations):
#             positions[i] = np.clip(positions[i], min_boundary_dist, L1 - min_boundary_dist)
#             for j in range(i):
#                 while np.linalg.norm(positions[i] - positions[j]) < min_dist:
#                     positions[i] = np.random.uniform(min_boundary_dist, L1 - min_boundary_dist, 2)
#         return positions
    
#     def update_positions(self, indices):
#         for i in indices:
#             attempts = 0
#             while attempts < 10:
#                 new_pos = self.positions[i] + np.random.uniform(-0.1, 0.1, self.positions[i].shape)
#                 new_pos = self.enforce_constraints(new_pos)
#                 if np.all(new_pos != self.positions[i]):
#                     self.positions[i] = new_pos
#                     break
#                 attempts += 1
    
#     def update_producers(self):
#         self.update_positions(self.producers)
    
#     def update_scroungers(self):
#         self.update_positions(self.scroungers)
    
#     def anti_predator_escape(self):
#         worst_index = np.argmin(self.fitness)
#         worst_pos = self.positions[worst_index]
#         for i in range(self.num_sparrows):
#             K = np.random.uniform(-1, 1)
#             if self.fitness[i] > self.best_fit:
#                 self.positions[i] += np.random.normal(0, 1, self.positions[i].shape) * (worst_pos - self.positions[i])
#             else:
#                 self.positions[i] += K * np.abs(self.positions[i] - self.best_pos)
#             self.positions[i] = self.enforce_constraints(self.positions[i])
    
#     def optimize(self):
#         for iter in range(self.max_iter):
#             print(f"Iteration {iter+1}/{self.max_iter}")  # Debug output
#             self.history.append((self.positions.copy(), self.best_pos.copy()))
#             self.update_roles()
#             self.update_producers()
#             self.update_scroungers()
#             self.anti_predator_escape()
#             self.fitness = np.array([coverage_ratio(pos, self.area_size, self.grid_size, self.Rs) for pos in self.positions])
#             best_index = np.argmax(self.fitness)
#             if self.fitness[best_index] > self.best_fit:
#                 self.best_fit = self.fitness[best_index]
#                 self.best_pos = self.positions[best_index]
#         return self.best_pos, self.best_fit
    
#     def visualize_coverage(self):
#         L1, L2 = self.area_size
#         m, n = self.grid_size
#         x_grid = np.linspace(0, L1, m)
#         y_grid = np.linspace(0, L2, n)
#         X, Y = np.meshgrid(x_grid, y_grid)
#         coverage_grid = np.zeros_like(X, dtype=int)
#         signal_grid = np.zeros_like(X, dtype=float)
    
#         for x, y in self.best_pos:
#             distances = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
#             coverage_grid |= is_covered(distances, self.Rs)
#             signal_grid += signal_strength(distances, self.Rs)
        
#         plt.figure(figsize=(8, 6))
#         plt.contourf(X, Y, signal_grid, levels=20, cmap='coolwarm', alpha=0.7)
#         plt.colorbar(label='Signal Strength')
#         plt.scatter(self.best_pos[:, 0], self.best_pos[:, 1], color='black', marker='x', s=100, label='Base Stations')
#         plt.title("Base Station Coverage and Signal Strength")
#         plt.xlabel("X Coordinate")
#         plt.ylabel("Y Coordinate")
#         plt.legend()
#         plt.show()

# print("Running SSA for Base Station Placement Optimization...")
# ssa = SparrowSearchAlgorithm()
# best_solution, best_value = ssa.optimize()
# print("Best Base Station Positions:\n", best_solution)
# print("Maximum Coverage Ratio:", best_value)
# ssa.visualize_coverage()

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

class EnhancedSparrowSearchAlgorithm:
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
    
    def initialize_positions(self):
        valid_positions = np.zeros((self.num_sparrows, self.num_stations, 2))
        min_dist = 3.1
        min_boundary_dist = 1
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
        min_dist = 3.1
        min_boundary_dist = 1
        L1, L2 = self.area_size
        
        for i in range(self.num_stations):
            positions[i] = np.clip(positions[i], min_boundary_dist, L1 - min_boundary_dist)
            for j in range(i):
                while np.linalg.norm(positions[i] - positions[j]) < min_dist:
                    positions[i] = np.random.uniform(min_boundary_dist, L1 - min_boundary_dist, 2)
        return positions
    
    def update_positions(self, indices):
        for i in indices:
            attempts = 0
            while attempts < 10:
                new_pos = self.positions[i] + np.random.uniform(-0.1, 0.1, self.positions[i].shape)
                new_pos = self.enforce_constraints(new_pos)
                if np.all(new_pos != self.positions[i]):
                    self.positions[i] = new_pos
                    break
                attempts += 1
    
    def update_producers(self):
        """Update positions for producers (discoverers)"""
        alpha = 0.8  # Constant for equation
        ST = 0.6     # Safety threshold
        Q = 0.1      # Step size
        R2 = np.random.rand()  # Random value for condition
        
        for i in self.producers:
            curr_iter = len(self.history) + 1
            if R2 < ST:
                # Cautious movement (equation 7, first part)
                exp_factor = np.exp(-i / (alpha * self.max_iter))
                self.positions[i] = self.positions[i] * exp_factor
            else:
                # Explore new areas (equation 7, second part)
                L = np.random.normal(0, 1, self.positions[i].shape)  # Random direction
                self.positions[i] = self.positions[i] + Q * L
                
            # Enforce constraints
            self.positions[i] = self.enforce_constraints(self.positions[i])
    
    def update_scroungers(self):
        """Update positions for scroungers (joiners)"""
        best_producer_idx = self.producers[0]  # Get best producer
        
        for i in self.scroungers:
            if np.random.rand() > 0.8:  # 20% chance for alert behavior
                # Move away from worst position (anti-predator behavior)
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
        worst_index = np.argmin(self.fitness)
        worst_pos = self.positions[worst_index]
        for i in range(self.num_sparrows):
            K = np.random.uniform(-1, 1)
            if self.fitness[i] > self.best_fit:
                self.positions[i] += np.random.normal(0, 1, self.positions[i].shape) * (worst_pos - self.positions[i])
            else:
                self.positions[i] += K * np.abs(self.positions[i] - self.best_pos)
            self.positions[i] = self.enforce_constraints(self.positions[i])
    
    def optimize(self):
        for iter in range(self.max_iter):
            print(f"Iteration {iter+1}/{self.max_iter}")  # Debug output
            self.history.append((self.positions.copy(), self.best_pos.copy(), self.best_fit))
            self.update_roles()
            self.update_producers()
            self.update_scroungers()
            self.anti_predator_escape()
            
            # Calculate new fitness values using the enhanced joint probability model
            self.fitness = np.array([coverage_ratio(pos, self.area_size, self.grid_size, 
                                                   self.Rs, self.Re, self.lambda_param) 
                                    for pos in self.positions])
            
            # Update best solution if improved
            best_index = np.argmax(self.fitness)
            if self.fitness[best_index] > self.best_fit:
                self.best_fit = self.fitness[best_index]
                self.best_pos = self.positions[best_index].copy()
                print(f"New best solution found! Coverage: {self.best_fit * 100:.2f}%")
            
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
    print("Running Enhanced SSA for Base Station Placement Optimization...")
    
    # Parameters
    area_size = (10, 10)
    grid_size = (50, 50)
    Rs = 2.2  # Coverage radius
    Re = 1.0  # Inner radius for probability model
    lambda_param = 1.0  # Parameter for probability exponential decay
    
    # Create and run the optimization
    ssa = EnhancedSparrowSearchAlgorithm(
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
    best_solution, best_value = ssa.optimize()
    
    # Print results
    print("\nOptimization Complete!")
    print("Best Base Station Positions:\n", best_solution)
    print(f"Maximum Coverage Ratio: {best_value * 100:.2f}%")
    
    # Visualize results
    ssa.visualize_coverage()
    ssa.visualize_convergence()