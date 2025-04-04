import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, distance
import matplotlib.patches as patches
import time  # For timing

# Global variables for shapes
pit_shape = None
avoidance_shape = None


def perception_probability(distance, Rs, Re=1.0, lambda_param=1.0):
    """
    Three-part perception probability model:
    p(zi,Hj) = 0, if d(zi,Hj) >= Rs
              = exp(-λ(Rs-d(zi,Hj))/(d(zi,Hj)-Rs-Re)), if Rs-Re < d(zi,Hj) < Rs
              = 1, if d(zi,Hj) <= Rs-Re
    """
    result = np.zeros_like(distance, dtype=float)
    mask_far = distance >= Rs
    mask_close = distance <= Rs - Re
    mask_middle = ~mask_far & ~mask_close
    
    result[mask_far] = 0
    result[mask_close] = 1
    
    # Handle the middle region with the exponential decay
    if np.any(mask_middle):
        result[mask_middle] = np.exp(-lambda_param * (Rs - distance[mask_middle]) / 
                                    (distance[mask_middle] - Rs - Re))
    return result

def joint_probability_vectorized(X, Y, stations, Rs, Re=1.0, lambda_param=1.0):
    """Calculate joint probability for grid of points"""
    joint_prob = np.ones_like(X, dtype=float)
    
    for i in range(len(stations)):
        distances = np.sqrt((X - stations[i, 0])**2 + (Y - stations[i, 1])**2)
        probs = perception_probability(distances, Rs, Re, lambda_param)
        joint_prob *= (1 - probs)
    
    return 1 - joint_prob

def is_inside_polygon(x, y, polygon):
    """Check if point (x,y) is inside polygon using ray casting algorithm"""
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def is_in_boundary_buffer(x, y, area_size, buffer_size=2.0):
    """Check if point (x,y) is within buffer_size distance from boundary"""
    L1, L2 = area_size
    return (x < buffer_size or x > L1 - buffer_size or 
            y < buffer_size or y > L2 - buffer_size)

def random_walk_normalization(X, a, b, c, d):
    """
    Implements the random walk normalization as described in equation 12:
    X^t = ((X^t - a_i) / (d_i - c_i)) * (b_i - a_i) + c_i
    """
    return (b - a) * ((X - c) / (d - c)) + a

# Create a vectorized version for points in a grid
def create_polygon_mask(X, Y, polygon):
    """Create a mask for points inside the polygon"""
    mask = np.zeros_like(X, dtype=bool)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            mask[i, j] = is_inside_polygon(X[i, j], Y[i, j], polygon)
    return mask

# Create a mask specifically for boundary buffer
def create_boundary_mask(X, Y, area_size, buffer_size=2.0):
    """Create a mask for points in boundary buffer zone"""
    mask = np.zeros_like(X, dtype=bool)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            mask[i, j] = is_in_boundary_buffer(X[i, j], Y[i, j], area_size, buffer_size)
    return mask

def generate_irregular_shape(center, num_points=10, scale=1.5, seed=None):
    """Generate an irregular convex polygon around a center point"""
    if seed is not None:
        np.random.seed(seed)
    
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    radii = scale * (1 + 0.3 * np.random.randn(num_points))
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    points = np.column_stack([x, y])
    hull = ConvexHull(points)
    return points[hull.vertices]

def coverage_ratio(base_stations, area_size, grid_size, Rs, Re=1.0, lambda_param=1.0, 
                   pit_mask=None, avoidance_mask=None, boundary_mask=None, 
                   X=None, Y=None, weight_pit=3.0):
    """Calculate coverage ratio with probabilistic model and priority areas"""
    global pit_shape, avoidance_shape  # Use the global variables
    
    L1, L2 = area_size
    m, n = grid_size
    
    # Convert base_stations to numpy array for vectorized operations
    base_stations_array = np.array(base_stations)
    
    # Calculate joint probability coverage
    prob_grid = joint_probability_vectorized(X, Y, base_stations_array, Rs, Re, lambda_param)
    
    # Calculate specific coverage metrics
    total_points = X.size
    total_coverage = np.sum(prob_grid) / total_points
    
    # Pit coverage (this is what we want to maximize)
    if pit_mask is not None:
        pit_points = np.sum(pit_mask)
        pit_coverage = np.sum(prob_grid[pit_mask]) / pit_points if pit_points > 0 else 0
    else:
        pit_coverage = 0
    
    # Calculate station placement penalty
    placement_penalty = 0
    for x, y in base_stations:
        # Check if station is in restricted area (pit) - major violation
        if is_inside_polygon(x, y, pit_shape):
            placement_penalty += 10.0
        # Check if station is in avoidance area - minor violation
        elif is_inside_polygon(x, y, avoidance_shape) and not is_inside_polygon(x, y, pit_shape):
            placement_penalty += 0.2
        # Check if station is in boundary buffer - minor violation
        elif is_in_boundary_buffer(x, y, area_size):
            placement_penalty += 5.0
    
    # Normalize placement penalty
    if len(base_stations) > 0:
        placement_penalty /= len(base_stations)
    
    # Calculate clustering penalty - reduced for more overlap allowed
    clustering_penalty = calculate_clustering_penalty(base_stations, Rs, threshold=1.0)
    
    # Combined fitness formula: prioritize pit coverage, penalize invalid placements
    return (0.7 * total_coverage + 1.2 * pit_coverage) - (1 * placement_penalty) - (2 * clustering_penalty)

def calculate_clustering_penalty(base_stations, Rs, threshold=1.0):
    # Increase the penalty weight for very close stations
    if len(base_stations) <= 1:
        return 0
    
    distances = distance.pdist(base_stations)
    
    # Create stronger penalties for stations that are very close
    # Use exponential penalty that grows as stations get closer
    close_distances = distances[distances < threshold * Rs]
    if len(close_distances) == 0:
        return 0
    
    # Apply exponential penalty that increases dramatically for very close stations
    penalty_factors = np.exp(threshold * Rs - close_distances) - 1
    return np.sum(penalty_factors) / (len(base_stations) * (len(base_stations) - 1) / 2)

class SparrowSearchAlgorithm:
    def __init__(self, num_sparrows=30, num_stations=5, area_size=(10, 10), grid_size=(50, 50), 
                 Rs=2.0, Re=1.5, lambda_param=1.0, max_iter=100, seed=42, early_stop_iter=20, boundary_buffer=2.0):
        global pit_shape, avoidance_shape  # Use the global variables
        
        self.num_sparrows = num_sparrows
        self.num_stations = num_stations
        self.area_size = area_size
        self.grid_size = grid_size
        self.Rs = Rs
        self.Re = Rs / 2  # Default Re is half of Rs
        self.lambda_param = lambda_param
        self.max_iter = max_iter
        self.early_stop_iter = early_stop_iter  # Early stopping parameter
        self.boundary_buffer = boundary_buffer
        
        # Set up pit and avoidance areas
        self.pit_center = (area_size[0] / 2, area_size[1] / 2)
        np.random.seed(seed)  # For reproducibility
        self.pit_shape = generate_irregular_shape(self.pit_center, num_points=8, scale=1.5, seed=seed)
        self.avoidance_shape = generate_irregular_shape(self.pit_center, num_points=12, scale=2.5, seed=seed)
        
        # Assign to global variables
        pit_shape = self.pit_shape
        avoidance_shape = self.avoidance_shape
        
        # Pre-compute grid and masks
        L1, L2 = self.area_size
        m, n = self.grid_size
        self.x_grid = np.linspace(0, L1, m)
        self.y_grid = np.linspace(0, L2, n)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)
        
        # Create masks for pit, avoidance areas, and boundary buffer
        self.pit_mask = create_polygon_mask(self.X, self.Y, self.pit_shape)
        self.avoidance_mask = create_polygon_mask(self.X, self.Y, self.avoidance_shape)
        self.boundary_mask = create_boundary_mask(self.X, self.Y, self.area_size, self.boundary_buffer)
        self.combined_avoidance_mask = self.avoidance_mask | self.boundary_mask
        
        # Initialize positions and fitness
        self.positions = self.initialize_positions()
        
        # Adaptive grid resolution - start with lower resolution for faster initial convergence
        self.current_grid_size = (min(25, grid_size[0]), min(25, grid_size[1]))
        self.current_X, self.current_Y = np.meshgrid(
            np.linspace(0, L1, self.current_grid_size[0]),
            np.linspace(0, L2, self.current_grid_size[1])
        )
        self.current_pit_mask = create_polygon_mask(self.current_X, self.current_Y, self.pit_shape)
        self.current_avoidance_mask = create_polygon_mask(self.current_X, self.current_Y, self.avoidance_shape)
        self.current_boundary_mask = create_boundary_mask(self.current_X, self.current_Y, self.area_size, self.boundary_buffer)
        self.current_combined_avoidance_mask = self.current_avoidance_mask | self.current_boundary_mask
        
        # Calculate initial fitness
        self.fitness = np.array([
            coverage_ratio(
                pos, area_size, self.current_grid_size, Rs, Re, lambda_param,
                self.current_pit_mask, self.current_avoidance_mask,
                self.current_boundary_mask, self.current_X, self.current_Y
            ) for pos in self.positions
        ])
        
        # Keep track of best solution
        self.best_pos = self.positions[np.argmax(self.fitness)]
        self.best_fit = np.max(self.fitness)
        
        # Keep track of convergence
        self.convergence_curve = []
        self.no_improvement_count = 0
        
        # Create validation grid (higher resolution) for final evaluation
        self.final_grid_size = grid_size
        self.final_X, self.final_Y = self.X, self.Y
        self.final_pit_mask = self.pit_mask
        self.final_avoidance_mask = self.avoidance_mask
        self.final_boundary_mask = self.boundary_mask
        self.final_combined_avoidance_mask = self.combined_avoidance_mask
        
        # Parameters for random walk
        self.L1, self.L2 = area_size
        self.a = self.boundary_buffer  # Lower bound
        self.b = self.L1 - self.boundary_buffer  # Upper bound

    def initialize_positions(self):
        """Initialize sparrow positions with stations outside restricted area and focus on surrounding area"""
        positions = []
        min_dist = self.Rs * 1.5  # Minimum distance between stations
        min_boundary_dist = 3  # Minimum distance from boundaries
        
        for _ in range(self.num_sparrows):
            valid_positions = []
            
            # Try to place stations outside all avoidance areas (including boundary buffer)
            attempts = 0
            max_attempts = 1000  # Prevent infinite loop
            
            while len(valid_positions) < min(3, self.num_stations) and attempts < max_attempts:
                attempts += 1
                
                # Select positions with min_boundary_dist from edges
                x = np.random.uniform(min_boundary_dist, self.area_size[0] - min_boundary_dist)
                y = np.random.uniform(min_boundary_dist, self.area_size[1] - min_boundary_dist)
                
                # Ensure the point is not in any restricted area
                if (not is_inside_polygon(x, y, self.pit_shape) and 
                    not is_inside_polygon(x, y, self.avoidance_shape)):
                    
                    # Apply minimum distance constraint between stations
                    if not valid_positions or all(np.sqrt((x - vx)**2 + (y - vy)**2) >= min_dist 
                        for vx, vy in valid_positions):
                        valid_positions.append([x, y])
                
                # If we've tried too many times, just pick a random point outside the pit
                if attempts >= max_attempts and len(valid_positions) < 1:
                    x = np.random.uniform(min_boundary_dist, self.area_size[0] - min_boundary_dist)
                    y = np.random.uniform(min_boundary_dist, self.area_size[1] - min_boundary_dist)
                    if not is_inside_polygon(x, y, self.pit_shape):
                        valid_positions.append([x, y])
            
            # Place remaining stations with more relaxed constraints (outside pit, can be in avoidance)
            # But still maintain minimum distance between stations and from boundaries
            attempts = 0
            while len(valid_positions) < self.num_stations and attempts < max_attempts:
                attempts += 1
                x = np.random.uniform(min_boundary_dist, self.area_size[0] - min_boundary_dist)
                y = np.random.uniform(min_boundary_dist, self.area_size[1] - min_boundary_dist)
                
                if not is_inside_polygon(x, y, self.pit_shape):
                    # Apply minimum distance constraint between stations
                    if all(np.sqrt((x - vx)**2 + (y - vy)**2) >= min_dist for vx, vy in valid_positions):
                        valid_positions.append([x, y])
            
            # If we still don't have enough stations after all attempts, add them anyway with just pit constraint
            # (This is a fallback to prevent getting stuck)
            while len(valid_positions) < self.num_stations:
                x = np.random.uniform(min_boundary_dist, self.area_size[0] - min_boundary_dist)
                y = np.random.uniform(min_boundary_dist, self.area_size[1] - min_boundary_dist)
                if not is_inside_polygon(x, y, self.pit_shape):
                    valid_positions.append([x, y])
            
            positions.append(valid_positions)
        
        return np.array(positions)

    def update_roles(self):
        """Update producer and scrounger roles based on fitness"""
        sorted_indices = np.argsort(-self.fitness)
        num_producers = max(1, int(0.2 * self.num_sparrows))
        self.producers = sorted_indices[:num_producers]
        self.scroungers = sorted_indices[num_producers:]
    
    def update_producers(self, iteration):
        """Update producer positions with adaptive step size"""
        alpha = 0.8  # Constant for equation
        ST = 0.6     # Safety threshold
        step_size = 0.2 * (1 - iteration / self.max_iter)
        R2 = np.random.rand()  # Random value for condition
        
        for i in self.producers:
            if R2 < ST:
                # Cautious movement - use exp factor as in equation 7
                exp_factor = np.exp(-i / (alpha * self.max_iter))
                for j in range(len(self.positions[i])):
                    self.positions[i][j] *= exp_factor
            else:
                # Explore new areas with random step
                for j in range(len(self.positions[i])):
                    # Update with random movement
                    self.positions[i][j] += np.random.uniform(-step_size, step_size, 2)
            
            # Ensure within bounds and not in pit
            for j in range(len(self.positions[i])):
                # Ensure within bounds
                self.positions[i][j] = np.clip(self.positions[i][j], 0, self.area_size[0])
                
                # Ensure not in restricted area (pit)
                attempts = 0
                while is_inside_polygon(self.positions[i][j][0], self.positions[i][j][1], self.pit_shape) and attempts < 5:
                    self.positions[i][j] += np.random.uniform(-step_size, step_size, 2)
                    self.positions[i][j] = np.clip(self.positions[i][j], 0, self.area_size[0])
                    attempts += 1
                
                # If still in pit after attempts, place randomly outside pit
                if is_inside_polygon(self.positions[i][j][0], self.positions[i][j][1], self.pit_shape):
                    valid = False
                    while not valid:
                        self.positions[i][j] = np.random.uniform(0, self.area_size[0], 2)
                        valid = not is_inside_polygon(self.positions[i][j][0], self.positions[i][j][1], self.pit_shape)
    
    def update_scroungers(self):
        """Update scrounger positions by following producers"""
        best_producer_idx = self.producers[0]  # Get best producer
        
        for i in self.scroungers:
            if np.random.rand() > 0.8:  # 20% chance for alert behavior
                # Move away from worst position (anti-predator)
                worst_idx = np.argmin(self.fitness)
                for j in range(len(self.positions[i])):
                    K = np.random.uniform(-1, 1)
                    self.positions[i][j] += K * np.abs(self.positions[i][j] - self.positions[worst_idx][j])
            else:
                # Normal joiner behavior - follow best producer
                for j in range(len(self.positions[i])):
                    A = np.random.randn(2)  # Random coefficient vector
                    self.positions[i][j] += np.random.rand() * A * (self.positions[best_producer_idx][j] - self.positions[i][j])
            
            # Ensure within bounds and not in pit
            for j in range(len(self.positions[i])):
                # Ensure within bounds
                self.positions[i][j] = np.clip(self.positions[i][j], 0, self.area_size[0])
                
                # Ensure not in restricted area (pit)
                if is_inside_polygon(self.positions[i][j][0], self.positions[i][j][1], self.pit_shape):
                    valid = False
                    while not valid:
                        self.positions[i][j] = np.random.uniform(0, self.area_size[0], 2)
                        valid = not is_inside_polygon(self.positions[i][j][0], self.positions[i][j][1], self.pit_shape)
    
    def apply_random_walk(self, index):
        """Implement random walk as described in equations 10-11"""
        # Generate random binary sequence
        steps = 100
        r = np.random.rand(steps) > 0.5
        r = 2 * r.astype(int) - 1  # Convert to -1 or 1
        
        # Calculate cumulative sum for the walk
        walk_x = np.zeros(steps + 1)
        walk_y = np.zeros(steps + 1)
        walk_x[1:] = np.cumsum(r)
        walk_y[1:] = np.cumsum(np.random.rand(steps) > 0.5) * 2 - 1
        
        # Get min/max for normalization
        min_x, max_x = np.min(walk_x), np.max(walk_x)
        min_y, max_y = np.min(walk_y), np.max(walk_y)
        
        for j in range(self.num_stations):
            # Get a random point from the walk
            idx = np.random.randint(0, steps)
            
            # Apply normalized position using equation 12
            pos_x = self.positions[index][j][0]
            pos_y = self.positions[index][j][1]
            
            # Normalize and apply the walk
            new_x = random_walk_normalization(walk_x[idx], self.a, self.b, min_x, max_x)
            new_y = random_walk_normalization(walk_y[idx], self.a, self.b, min_y, max_y)
            
            # Combine current position with random walk influence
            self.positions[index][j][0] = 0.7 * pos_x + 0.3 * new_x
            self.positions[index][j][1] = 0.7 * pos_y + 0.3 * new_y
            
            # Ensure not in pit
            if is_inside_polygon(self.positions[index][j][0], self.positions[index][j][1], self.pit_shape):
                valid = False
                while not valid:
                    self.positions[index][j] = np.random.uniform(0, self.area_size[0], 2)
                    valid = not is_inside_polygon(self.positions[index][j][0], self.positions[index][j][1], self.pit_shape)
    
    def random_walk(self, iteration):
        """Perform random walk for exploration - using new implementation"""
        # Only do this 20% of the time
        if np.random.rand() < 0.2:
            # Indices of sparrows to apply random walk (20%)
            walk_indices = np.random.choice(
                self.num_sparrows, 
                size=max(1, int(0.2 * self.num_sparrows)), 
                replace=False
            )
            
            for i in walk_indices:
                self.apply_random_walk(i)
    
    def anti_predator_escape(self):
        """Implement anti-predator behavior for the lowest-performing sparrows"""
        # Select 10% worst performers to apply escape
        num_escape = max(1, int(0.1 * self.num_sparrows))
        escape_indices = np.argsort(self.fitness)[:num_escape]
        
        for i in escape_indices:
            for j in range(self.num_stations):
                if np.random.rand() < 0.8:  # 80% chance to escape
                    # Random jump to a new location
                    self.positions[i][j] = np.random.uniform(0, self.area_size[0], 2)
                else:
                    # Move towards global best
                    self.positions[i][j] += np.random.normal(0, 0.5, 2) * (self.best_pos[j] - self.positions[i][j])
                
                # Ensure within bounds
                self.positions[i][j] = np.clip(self.positions[i][j], 0, self.area_size[0])
                
                # Ensure not in restricted area (pit)
                if is_inside_polygon(self.positions[i][j][0], self.positions[i][j][1], self.pit_shape):
                    valid = False
                    while not valid:
                        self.positions[i][j] = np.random.uniform(0, self.area_size[0], 2)
                        valid = not is_inside_polygon(self.positions[i][j][0], self.positions[i][j][1], self.pit_shape)
    
    def adjust_grid_resolution(self, iteration):
        """Dynamically adjust grid resolution based on iteration"""
        if iteration > self.max_iter // 2:
            # Switch to higher resolution for fine-tuning
            self.current_grid_size = (
                min(self.current_grid_size[0] * 2, self.final_grid_size[0]),
                min(self.current_grid_size[1] * 2, self.final_grid_size[1])
            )
            L1, L2 = self.area_size
            self.current_X, self.current_Y = np.meshgrid(
                np.linspace(0, L1, self.current_grid_size[0]),
                np.linspace(0, L2, self.current_grid_size[1])
            )
            self.current_pit_mask = create_polygon_mask(self.current_X, self.current_Y, self.pit_shape)
            self.current_avoidance_mask = create_polygon_mask(self.current_X, self.current_Y, self.avoidance_shape)
            self.current_boundary_mask = create_boundary_mask(self.current_X, self.current_Y, self.area_size, self.boundary_buffer)
            self.current_combined_avoidance_mask = self.current_avoidance_mask | self.current_boundary_mask
    
    def optimize(self):
        """Main optimization loop with early stopping and adaptive resolution"""
        global pit_shape, avoidance_shape  # Make these accessible to other functions
        pit_shape = self.pit_shape
        avoidance_shape = self.avoidance_shape
        
        start_time = time.time()
        
        for iter in range(self.max_iter):
            # Print progress less frequently
            if (iter + 1) % 20 == 0:
                elapsed = time.time() - start_time
                print(f"Iteration {iter + 1}/{self.max_iter}, Best Fitness: {self.best_fit:.4f}, Time: {elapsed:.2f}s")
            
            # Adjust grid resolution
            if (iter + 1) % 40 == 0:
                self.adjust_grid_resolution(iter)
            
            self.update_roles()
            self.update_producers(iter)
            self.update_scroungers()
            
            # Less frequent random walk and anti-predator behaviors
            if (iter + 1) % 3 == 0:
                self.random_walk(iter)
            if (iter + 1) % 5 == 0:
                self.anti_predator_escape()
            
            # Update fitness - vectorized calculation
            prev_best_fit = self.best_fit
            
            # Calculate fitness for all solutions
            for i in range(self.num_sparrows):
                self.fitness[i] = coverage_ratio(
                    self.positions[i], self.area_size, self.current_grid_size, self.Rs, self.Re, self.lambda_param,
                    self.current_pit_mask, self.current_avoidance_mask,
                    self.current_boundary_mask, self.current_X, self.current_Y
                )
            
            # Update best solution
            best_index = np.argmax(self.fitness)
            if self.fitness[best_index] > self.best_fit:
                self.best_fit = self.fitness[best_index]
                self.best_pos = self.positions[best_index].copy()
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
            
            # Record convergence
            self.convergence_curve.append(self.best_fit)
            
            # Early stopping condition
            if self.no_improvement_count >= self.early_stop_iter:
                print(f"Early stopping at iteration {iter + 1} due to no improvement")
                break
        
        # Final evaluation with high resolution grid
        print("Performing final evaluation with high resolution grid...")
        final_fitness = coverage_ratio(
            self.best_pos, self.area_size, self.final_grid_size, self.Rs, self.Re, self.lambda_param,
            self.final_pit_mask, self.final_avoidance_mask,
            self.final_boundary_mask, self.final_X, self.final_Y
        )
        print(f"Final fitness: {final_fitness:.4f}")
        
        total_time = time.time() - start_time
        print(f"Total optimization time: {total_time:.2f} seconds")
        
        return self.best_pos, final_fitness

    def visualize_coverage(self):
        """Visualize the coverage with improved preferences"""
        L1, L2 = self.area_size
        m, n = self.final_grid_size
        
        # Reuse final grid for visualization
        X, Y = self.final_X, self.final_Y
        
        # Calculate joint probability coverage directly
        base_stations_array = np.array(self.best_pos)
        prob_grid = joint_probability_vectorized(X, Y, base_stations_array, self.Rs, self.Re, self.lambda_param)
        
        plt.figure(figsize=(12, 9))
        
        # Plot signal strength
        contour = plt.contourf(X, Y, prob_grid, levels=20, cmap='viridis', alpha=0.7)
        cbar = plt.colorbar(contour, label='Coverage Probability')
        
        # Plot boundary avoidance area
        boundary_rect = patches.Rectangle((0, 0), L1, L2, linewidth=2, 
                                        edgecolor='red', facecolor='none')
        inner_rect = patches.Rectangle((self.boundary_buffer, self.boundary_buffer), 
                                    L1-2*self.boundary_buffer, L2-2*self.boundary_buffer, 
                                    linewidth=2, edgecolor='red', facecolor='none', 
                                    linestyle='--')
        plt.gca().add_patch(boundary_rect)
        plt.gca().add_patch(inner_rect)
        
        # Shade boundary buffer zone
        plt.fill_between([0, L1], [0, 0], [self.boundary_buffer, self.boundary_buffer], 
                        color='gray', alpha=0.2)
        plt.fill_between([0, L1], [L2, L2], [L2-self.boundary_buffer, L2-self.boundary_buffer], 
                        color='gray', alpha=0.2)
        plt.fill_between([0, self.boundary_buffer], [self.boundary_buffer, self.boundary_buffer], 
                        [L2-self.boundary_buffer, L2-self.boundary_buffer], color='gray', alpha=0.2)
        plt.fill_between([L1-self.boundary_buffer, L1], [self.boundary_buffer, self.boundary_buffer], 
                        [L2-self.boundary_buffer, L2-self.boundary_buffer], color='gray', alpha=0.2)
        
        # Plot avoidance area with shading
        plt.fill(self.avoidance_shape[:, 0], self.avoidance_shape[:, 1], 
                color='gray', alpha=0.2, label="Avoidance Area")
        plt.plot(self.avoidance_shape[:, 0], self.avoidance_shape[:, 1], 
                color='gray', linestyle="--", alpha=0.8)
        
        # Plot restricted area (pit)
        plt.fill(self.pit_shape[:, 0], self.pit_shape[:, 1], 
                color='black', alpha=0.7, label="Pit Area (Target Coverage)")
        
        # Plot base stations
        plt.scatter(self.best_pos[:, 0], self.best_pos[:, 1], 
                    color='red', marker='x', s=150, linewidth=3, label='Base Stations')
        
        # Draw coverage radius
        for x, y in self.best_pos:
            circle = plt.Circle((x, y), self.Rs, color='blue', fill=False, 
                            linestyle='--', alpha=0.5)
            plt.gca().add_patch(circle)
        
        # Add labels and title
        plt.title("Base Station Coverage with Target Pit Area and Boundary Buffer", fontsize=16)
        plt.xlabel("X Coordinate", fontsize=12)
        plt.ylabel("Y Coordinate", fontsize=12)
        plt.legend(loc='upper right', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Plot convergence curve
        # plt.figure(figsize=(8, 5))
        # plt.plot(np.arange(1, len(self.convergence_curve) + 1), self.convergence_curve, 'b-', linewidth=2)
        # plt.xlabel('Iteration', fontsize=12)
        # plt.ylabel('Fitness Value', fontsize=12)
        # plt.title('Convergence Curve', fontsize=14)
        # plt.grid(True, linestyle='--', alpha=0.3)
        # plt.tight_layout()
        # plt.show()
        
        # Calculate and display coverage metrics
        self.calculate_coverage_metrics()
    def calculate_coverage_metrics(self):
        """Calculate and display coverage metrics without using is_covered_vectorized"""
        # Reuse final grid for metrics
        X, Y = self.final_X, self.final_Y
        
        # Calculate coverage grid directly
        coverage_grid = np.zeros_like(X, dtype=bool)
        
        # For each point in the grid, check coverage from all stations
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for x, y in self.best_pos:
                    # Calculate distance to station
                    distance = np.sqrt((X[i, j] - x)**2 + (Y[i, j] - y)**2)
                    # If within coverage radius, mark as covered
                    if distance <= self.Rs:
                        coverage_grid[i, j] = True
                        break  # Once covered by one station, no need to check others
        
        # Calculate pit area coverage (target area)
        pit_points = np.sum(self.final_pit_mask)
        pit_covered = np.sum(coverage_grid & self.final_pit_mask)
        
        # Calculate avoidance area coverage
        avoidance_points = np.sum(self.final_avoidance_mask & ~self.final_pit_mask)
        avoidance_covered = np.sum(coverage_grid & self.final_avoidance_mask & ~self.final_pit_mask)
        
        # Calculate normal area coverage (outside both pit and avoidance)
        normal_mask = ~(self.final_pit_mask | self.final_avoidance_mask)
        normal_points = np.sum(normal_mask)
        normal_covered = np.sum(coverage_grid & normal_mask)
        
        total_points = X.size
        total_covered = np.sum(coverage_grid)
        
        print("\nCoverage Metrics:")
        print(f"Total Area Coverage: {total_covered / total_points * 100:.2f}%")
        if pit_points > 0:
            print(f"Pit Area Coverage (TARGET): {pit_covered / pit_points * 100:.2f}%")
        if avoidance_points > 0:
            print(f"Avoidance Area Coverage: {avoidance_covered / avoidance_points * 100:.2f}%")
        if normal_points > 0:
            print(f"Normal Area Coverage: {normal_covered / normal_points * 100:.2f}%")
        
        # Check for stations in restricted areas
        stations_in_pit = 0
        stations_in_avoidance = 0
        
        for x, y in self.best_pos:
            if is_inside_polygon(x, y, self.pit_shape):
                stations_in_pit += 1
            elif is_inside_polygon(x, y, self.avoidance_shape):
                stations_in_avoidance += 1
        
        print(f"\nStation Placement:")
        print(f"Stations in Pit Area: {stations_in_pit} (Should be 0)")
        print(f"Stations in Avoidance Area: {stations_in_avoidance}")
        
        # Calculate overlap between stations
        overlap_count = 0
        for i in range(len(self.best_pos)):
            for j in range(i+1, len(self.best_pos)):
                dist = np.sqrt(np.sum((self.best_pos[i] - self.best_pos[j])**2))
                if dist < 2 * self.Rs:
                    overlap_count += 1
        
        print(f"Station Overlap Count: {overlap_count}")

# Run the algorithm with optimized parameters
# Run the algorithm with parameter optimization
if __name__ == "__main__":
    print("Running Parameter Optimization for Base Station Placement...")
    
    # Fixed parameters
    area_size = (10, 10)
    grid_size = (100, 100)
    max_iter = 200
    early_stop = 50
    num_sparrows = 50
    
    # Parameters to optimize
    station_range = range(2, 7)  # Try 2 to 7 stations
    rs_values = [round(1.0 + i * 0.2, 1) for i in range(21)]  # 1.0 to 5.0 with step 0.1
    
    best_overall_value = -float('inf')
    best_overall_solution = None
    best_overall_params = None
    
    results = []
    
    print("Starting parameter sweep...")
    print(f"Testing {len(station_range)} station values and {len(rs_values)} radius values")
    print(f"Total configurations to test: {len(station_range) * len(rs_values)}")
    
    # Loop through all combinations
    for num_stations in station_range:
        for rs in rs_values:
            print(f"\nTesting with {num_stations} stations and Rs = {rs}")
            
            # Create and run the algorithm with these parameters
            ssa = SparrowSearchAlgorithm(
                num_sparrows=num_sparrows,
                num_stations=num_stations,
                area_size=area_size,
                grid_size=grid_size,
                Rs=rs,
                max_iter=max_iter,
                seed=42,
                early_stop_iter=early_stop
            )
            
            solution, value = ssa.optimize()
            
            # Store results for this configuration
            results.append({
                'num_stations': num_stations,
                'Rs': rs,
                'fitness': value,
                'solution': solution
            })
            
            print(f"Configuration with {num_stations} stations and Rs = {rs}: {value:.4f}")
            
            # Update best overall solution if better
            if value > best_overall_value:
                best_overall_value = value
                best_overall_solution = solution.copy()
                best_overall_params = {'num_stations': num_stations, 'Rs': rs}
    
    # Print summary of all configurations
    print("\n===== Parameter Optimization Results =====")
    print("Top 10 configurations:")
    for result in sorted(results, key=lambda x: x['fitness'], reverse=True)[:10]:
        print(f"Stations: {result['num_stations']}, Rs: {result['Rs']}, Fitness: {result['fitness']:.4f}")
    
    print("\n===== Best Configuration =====")
    print(f"Best Parameters: {best_overall_params['num_stations']} stations with Rs = {best_overall_params['Rs']}")
    print(f"Best Fitness Value: {best_overall_value:.4f}")
    print("Best Base Station Positions:")
    for i, (x, y) in enumerate(best_overall_solution):
        print(f"Station {i+1}: ({x:.4f}, {y:.4f})")
    
    # Save results to file
    import json
    with open('parameter_optimization_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = []
        for result in results:
            serializable_result = {
                'num_stations': result['num_stations'],
                'Rs': result['Rs'],
                'fitness': result['fitness'],
                'solution': result['solution'].tolist()
            }
            serializable_results.append(serializable_result)
        json.dump(serializable_results, f, indent=2)
    
    print("\nResults saved to parameter_optimization_results.json")
    
    # Run final visualization with best parameters
    print("\nGenerating visualization with best configuration...")
    final_ssa = SparrowSearchAlgorithm(
        num_sparrows=num_sparrows,
        num_stations=best_overall_params['num_stations'],
        area_size=area_size,
        grid_size=grid_size,
        Rs=best_overall_params['Rs'],
        max_iter=max_iter,
        seed=42,
        early_stop_iter=early_stop
    )
    
    # Set the best solution and run visualization
    final_ssa.best_pos = best_overall_solution
    final_ssa.visualize_coverage()