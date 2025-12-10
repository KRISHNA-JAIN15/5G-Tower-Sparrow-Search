import numpy as np
import matplotlib.pyplot as plt

def is_covered(distance, Rs):
    return distance <= Rs

def signal_strength(distance, Rs):
    return np.exp(-distance / Rs)

def coverage_ratio(base_stations, area_size, grid_size, Rs):
    L1, L2 = area_size
    m, n = grid_size
    x_grid = np.linspace(0, L1, m)
    y_grid = np.linspace(0, L2, n)
    X, Y = np.meshgrid(x_grid, y_grid)
    coverage_grid = np.zeros_like(X, dtype=int)
    
    for x, y in base_stations:
        distances = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
        coverage_grid |= is_covered(distances, Rs)
    
    total_covered = np.sum(coverage_grid)
    total_points = m * n
    print(f"Total Area Coverage: {total_covered / total_points * 100:.2f}%")
    
    return total_covered / total_points

def enforce_constraints(positions, area_size, min_dist=3.1, min_boundary_dist=1.2):
    L1, L2 = area_size
    positions = np.clip(positions, min_boundary_dist, L1 - min_boundary_dist)
    
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            while np.linalg.norm(positions[i] - positions[j]) < min_dist:
                positions[j] += np.random.uniform(-1, 1, 2)
                positions[j] = np.clip(positions[j], min_boundary_dist, L1 - min_boundary_dist)
    
    return positions

class SparrowSearchAlgorithm:
    def __init__(self, num_sparrows=30, num_stations=5, area_size=(10, 10), grid_size=(50, 50), Rs=2.2, max_iter=100):
        self.num_sparrows = num_sparrows
        self.num_stations = num_stations
        self.area_size = area_size
        self.grid_size = grid_size
        self.Rs = Rs
        self.max_iter = max_iter
        self.positions = np.random.uniform(0, area_size[0], (num_sparrows, num_stations, 2))
        self.positions = np.array([enforce_constraints(pos, area_size) for pos in self.positions])
        self.fitness = np.array([coverage_ratio(pos, area_size, grid_size, Rs) for pos in self.positions])
        self.best_pos = self.positions[np.argmax(self.fitness)]
        self.best_fit = np.max(self.fitness)
        self.history = []
    
    def update_roles(self):
        sorted_indices = np.argsort(-self.fitness)
        num_producers = max(1, int(0.2 * self.num_sparrows))
        self.producers = sorted_indices[:num_producers]
        self.scroungers = sorted_indices[num_producers:]
    
    def update_producers(self):
        for i in self.producers:
            self.positions[i] += np.random.uniform(-0.1, 0.1, self.positions[i].shape)
            self.positions[i] = enforce_constraints(self.positions[i], self.area_size)
    
    def update_scroungers(self):
        for i in self.scroungers:
            A = np.random.choice([-1, 1], size=self.positions[i].shape)
            Xp = self.positions[self.producers[np.argmax(self.fitness[self.producers])]]
            self.positions[i] += A * np.abs(self.positions[i] - Xp)
            self.positions[i] = enforce_constraints(self.positions[i], self.area_size)
    
    def random_walk(self, iteration):
        steps = np.cumsum(2 * (np.random.rand(self.max_iter) > 0.5) - 1)
        step_size = (steps[iteration] - np.min(steps)) / (np.max(steps) - np.min(steps) + 1e-9)
        step_size *= 0.5 * np.exp(-iteration / self.max_iter)
        
        for i in range(self.num_sparrows):
            self.positions[i] += step_size * np.random.uniform(-1, 1, self.positions[i].shape)
            self.positions[i] = enforce_constraints(self.positions[i], self.area_size)
    
    def anti_predator_escape(self):
        worst_index = np.argmin(self.fitness)
        worst_pos = self.positions[worst_index]
        for i in range(self.num_sparrows):
            K = np.random.uniform(-1, 1)
            if self.fitness[i] > self.best_fit:
                self.positions[i] += np.random.normal(0, 1, self.positions[i].shape) * (worst_pos - self.positions[i])
            else:
                self.positions[i] += K * np.abs(self.positions[i] - self.best_pos)
            self.positions[i] = enforce_constraints(self.positions[i], self.area_size)
    
    def optimize(self):
        for iter in range(self.max_iter):
            self.history.append((self.positions.copy(), self.best_pos.copy()))
            self.update_roles()
            self.update_producers()
            self.update_scroungers()
            self.random_walk(iter)
            self.anti_predator_escape()
            self.fitness = np.array([coverage_ratio(pos, self.area_size, self.grid_size, self.Rs) for pos in self.positions])
            best_index = np.argmax(self.fitness)
            if self.fitness[best_index] > self.best_fit:
                self.best_fit = self.fitness[best_index]
                self.best_pos = self.positions[best_index]
        return self.best_pos, self.best_fit
    
    def visualize_coverage(self):
        L1, L2 = self.area_size
        m, n = self.grid_size
        x_grid = np.linspace(0, L1, m)
        y_grid = np.linspace(0, L2, n)
        X, Y = np.meshgrid(x_grid, y_grid)
        coverage_grid = np.zeros_like(X, dtype=int)

        for x, y in self.best_pos:
            distances = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
            coverage_grid |= is_covered(distances, self.Rs)

        plt.figure(figsize=(8, 6))
        plt.imshow(coverage_grid, extent=[0, L1, 0, L2], origin='lower', cmap='OrRd', alpha=0.6)
        plt.scatter(self.best_pos[:, 0], self.best_pos[:, 1], color='black', marker='x', s=100, label='Base Stations')
        plt.title("Base Station Coverage Area")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.show()

print("Running SSA for Base Station Placement Optimization...")
ssa = SparrowSearchAlgorithm()
best_solution, best_value = ssa.optimize()
print("Best Base Station Positions:\n", best_solution)
print("Maximum Coverage Ratio:", best_value)
ssa.visualize_coverage()