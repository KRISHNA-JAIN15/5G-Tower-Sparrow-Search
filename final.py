import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, distance
import matplotlib.patches as patches
from numba import jit, njit  # For Just-In-Time compilation
import time  # For timing

# Global variables for shapes
pit_shape = None
avoidance_shape = None

# JIT-compiled helper functions
@njit
def is_covered_vectorized(X, Y, station_x, station_y, Rs):
    """Vectorized version of coverage calculation"""
    distances = np.sqrt((X - station_x) ** 2 + (Y - station_y) ** 2)
    return distances <= Rs

@njit
def signal_strength_vectorized(X, Y, station_x, station_y, Rs):
    """Vectorized version of signal strength calculation"""
    distances = np.sqrt((X - station_x) ** 2 + (Y - station_y) ** 2)
    return np.exp(-distances / Rs)

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

def coverage_ratio(base_stations, area_size, grid_size, Rs, pit_mask, avoidance_mask, boundary_mask, X, Y, weight_pit=3.0):
    """Calculate coverage ratio that prioritizes pit area coverage while respecting placement constraints"""
    global pit_shape, avoidance_shape  # Use the global variables
    
    L1, L2 = area_size
    m, n = grid_size
    
    # Initialize coverage grid
    coverage_grid = np.zeros_like(X, dtype=bool)
    
    # Calculate coverage for each base station
    for x, y in base_stations:
        coverage_grid |= is_covered_vectorized(X, Y, x, y, Rs)
    
    # Calculate specific coverage metrics
    total_points = X.size
    total_covered = np.sum(coverage_grid)
    
    # Pit coverage (this is what we want to maximize)
    pit_points = np.sum(pit_mask)
    pit_covered = np.sum(coverage_grid & pit_mask) if pit_points > 0 else 0
    
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
            placement_penalty += 0.2
    
    # Normalize placement penalty
    if len(base_stations) > 0:
        placement_penalty /= len(base_stations)
    
    # Calculate clustering penalty - reduced for more overlap allowed
    clustering_penalty = calculate_clustering_penalty(base_stations, Rs, threshold=1.2)
    
    # Calculate overall fitness with high weight for pit coverage
    base_coverage = total_covered / total_points
    pit_coverage_ratio = pit_covered / pit_points if pit_points > 0 else 0
    
    # Combined fitness formula: prioritize pit coverage, penalize invalid placements
    return (0.3 * base_coverage + 0.7 * pit_coverage_ratio) - (0.5 * placement_penalty) - (0.15 * clustering_penalty)

def calculate_clustering_penalty(base_stations, Rs, threshold=1.2):
    """Calculate clustering penalty with adjustable threshold"""
    if len(base_stations) <= 1:
        return 0
    
    # Calculate pairwise distances between stations
    distances = distance.pdist(base_stations)
    
    # Identify close stations (closer than threshold * Rs)
    close_pairs = np.sum(distances < threshold * Rs)
    
    # Return penalty based on number of close pairs
    return close_pairs / (len(base_stations) * (len(base_stations) - 1) / 2)

class SparrowSearchAlgorithm:
    def __init__(self, num_sparrows=30, num_stations=5, area_size=(10, 10), grid_size=(50, 50), Rs=2, max_iter=100, 
                 seed=42, early_stop_iter=20, boundary_buffer=2.0, num_pits=4):
        
        self.num_sparrows = num_sparrows
        self.num_stations = num_stations
        self.area_size = area_size
        self.grid_size = grid_size
        self.Rs = Rs
        self.max_iter = max_iter
        self.early_stop_iter = early_stop_iter
        self.boundary_buffer = boundary_buffer
        
        # New parameters for multiple regions
        self.num_pits = num_pits
        # Set num_avoidance_regions equal to num_pits
        self.num_avoidance_regions = num_pits
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Generate pit areas first
        self.pit_centers, self.pits = self.generate_pit_regions(self.num_pits, pit_scale=1.5, min_distance=3.0)
        
        # Generate avoidance areas centered on the same locations as pits but with larger scale
        self.avoidance_regions = self.generate_avoidance_regions(self.pit_centers, avoidance_scale=2.5)
        
        # Pre-compute grid and masks
        L1, L2 = self.area_size
        m, n = self.grid_size
        self.x_grid = np.linspace(0, L1, m)
        self.y_grid = np.linspace(0, L2, n)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)
        
        # Create masks for pits, avoidance areas, and boundary buffer
        self.pit_masks = [create_polygon_mask(self.X, self.Y, pit) for pit in self.pits]
        self.combined_pit_mask = np.any(self.pit_masks, axis=0)
        
        self.avoidance_masks = [create_polygon_mask(self.X, self.Y, region) for region in self.avoidance_regions]
        self.combined_avoidance_mask = np.any(self.avoidance_masks, axis=0)
        
        self.boundary_mask = create_boundary_mask(self.X, self.Y, self.area_size, self.boundary_buffer)
        self.total_restricted_mask = self.combined_pit_mask | self.combined_avoidance_mask | self.boundary_mask
        
        # Initialize positions and fitness
        self.positions = self.initialize_positions()
        
        # Adaptive grid resolution - start with lower resolution for faster initial convergence
        self.current_grid_size = (min(25, grid_size[0]), min(25, grid_size[1]))
        self.current_X, self.current_Y = np.meshgrid(
            np.linspace(0, L1, self.current_grid_size[0]),
            np.linspace(0, L2, self.current_grid_size[1])
        )
        
        # Create current resolution masks
        self.current_pit_masks = [create_polygon_mask(self.current_X, self.current_Y, pit) for pit in self.pits]
        self.current_combined_pit_mask = np.any(self.current_pit_masks, axis=0)
        
        self.current_avoidance_masks = [create_polygon_mask(self.current_X, self.current_Y, region) for region in self.avoidance_regions]
        self.current_combined_avoidance_mask = np.any(self.current_avoidance_masks, axis=0)
        
        self.current_boundary_mask = create_boundary_mask(self.current_X, self.current_Y, self.area_size, self.boundary_buffer)
        self.current_total_restricted_mask = self.current_combined_pit_mask | self.current_combined_avoidance_mask | self.current_boundary_mask
        
        # Calculate initial fitness
        self.fitness = np.array([
            self.coverage_ratio_multi_region(pos) for pos in self.positions
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
        self.final_pit_masks = self.pit_masks
        self.final_combined_pit_mask = self.combined_pit_mask
        self.final_avoidance_masks = self.avoidance_masks
        self.final_combined_avoidance_mask = self.combined_avoidance_mask
        self.final_boundary_mask = self.boundary_mask
        self.final_total_restricted_mask = self.total_restricted_mask

    def generate_pit_regions(self, num_regions, pit_scale, min_distance):
        """Generate multiple non-overlapping pit regions and return their centers and shapes"""
        regions = []
        centers = []
        attempts = 0
        max_attempts = 100  # Avoid infinite loop
        
        while len(regions) < num_regions and attempts < max_attempts:
            # Generate random center point
            center = (
                np.random.uniform(self.boundary_buffer + pit_scale, self.area_size[0] - self.boundary_buffer - pit_scale),
                np.random.uniform(self.boundary_buffer + pit_scale, self.area_size[1] - self.boundary_buffer - pit_scale)
            )
            
            # Check if this center is far enough from existing regions
            if regions and any(np.sqrt((center[0] - c[0])**2 + (center[1] - c[1])**2) < min_distance for c in centers):
                attempts += 1
                continue
            
            # Generate irregular shape
            new_region = generate_irregular_shape(center, num_points=8, scale=pit_scale, seed=None)
            
            # Check if shape is within boundaries
            if np.any(new_region[:, 0] < self.boundary_buffer) or np.any(new_region[:, 0] > self.area_size[0] - self.boundary_buffer) or \
               np.any(new_region[:, 1] < self.boundary_buffer) or np.any(new_region[:, 1] > self.area_size[1] - self.boundary_buffer):
                attempts += 1
                continue
                
            # Add the shape and center to our collections
            regions.append(new_region)
            centers.append(center)
            attempts = 0  # Reset attempts counter after successful addition
            
        if len(regions) < num_regions:
            print(f"Warning: Could only generate {len(regions)} pit regions out of {num_regions} requested")
            
        return centers, regions

    def generate_avoidance_regions(self, centers, avoidance_scale):
        """Generate avoidance regions around the same centers as pit regions but with larger scale"""
        avoidance_regions = []
        
        for center in centers:
            # Generate irregular shape with larger scale around the same center
            avoidance_region = generate_irregular_shape(center, num_points=8, scale=avoidance_scale, seed=None)
            
            # Check if shape is within boundaries, if not, adjust
            if np.any(avoidance_region[:, 0] < 0):
                min_x = np.min(avoidance_region[:, 0])
                avoidance_region[:, 0] -= min_x - 0.1
            if np.any(avoidance_region[:, 0] > self.area_size[0]):
                max_x = np.max(avoidance_region[:, 0])
                avoidance_region[:, 0] -= (max_x - self.area_size[0] + 0.1)
            if np.any(avoidance_region[:, 1] < 0):
                min_y = np.min(avoidance_region[:, 1])
                avoidance_region[:, 1] -= min_y - 0.1
            if np.any(avoidance_region[:, 1] > self.area_size[1]):
                max_y = np.max(avoidance_region[:, 1])
                avoidance_region[:, 1] -= (max_y - self.area_size[1] + 0.1)
                
            avoidance_regions.append(avoidance_region)
            
        return avoidance_regions

    def initialize_positions(self):
        """Initialize sparrow positions with stations outside restricted areas"""
        positions = []
        for _ in range(self.num_sparrows):
            valid_positions = []
            
            # Try to place stations outside all restricted areas
            while len(valid_positions) < self.num_stations:
                # Select positions in the "safe" zone
                x = np.random.uniform(self.boundary_buffer, self.area_size[0] - self.boundary_buffer)
                y = np.random.uniform(self.boundary_buffer, self.area_size[1] - self.boundary_buffer)
                
                # Check if point is in any pit
                in_pit = any(is_inside_polygon(x, y, pit) for pit in self.pits)
                
                # Check if point is in any avoidance region
                in_avoidance = any(is_inside_polygon(x, y, region) for region in self.avoidance_regions)
                
                # Check if point is in boundary buffer
                in_buffer = is_in_boundary_buffer(x, y, self.area_size, self.boundary_buffer)
                
                # Ensure the point is not in any restricted area
                if not in_pit and not in_avoidance and not in_buffer:
                    # Ensure minimum distance between stations
                    if not valid_positions or all(np.sqrt((x - vx)**2 + (y - vy)**2) > 0.7 * self.Rs 
                          for vx, vy in valid_positions):
                        valid_positions.append([x, y])
                
                # If we've tried too many times, just pick a random point outside the pits
                if len(valid_positions) < 1:
                    x, y = np.random.uniform(0, self.area_size[0], 2)
                    if not any(is_inside_polygon(x, y, pit) for pit in self.pits):
                        valid_positions.append([x, y])
            
            positions.append(valid_positions)
        
        return np.array(positions)

    def coverage_ratio_multi_region(self, stations, use_current_grid=True):
        """Calculate coverage ratio with multiple regions"""
        if use_current_grid:
            X, Y = self.current_X, self.current_Y
            pit_mask = self.current_combined_pit_mask
            avoidance_mask = self.current_combined_avoidance_mask
            boundary_mask = self.current_boundary_mask
        else:
            X, Y = self.final_X, self.final_Y
            pit_mask = self.final_combined_pit_mask
            avoidance_mask = self.final_combined_avoidance_mask
            boundary_mask = self.final_boundary_mask
        
        # Calculate coverage grid
        coverage_grid = np.zeros_like(X, dtype=bool)
        for x, y in stations:
            coverage_grid |= is_covered_vectorized(X, Y, x, y, self.Rs)
        
        # Calculate pit coverage
        pit_points = np.sum(pit_mask)
        pit_covered = np.sum(coverage_grid & pit_mask)
        
        # Calculate avoidance area coverage (penalize)
        avoidance_points = np.sum(avoidance_mask)
        avoidance_covered = np.sum(coverage_grid & avoidance_mask)
        
        # Calculate total coverage excluding boundary
        valid_area = ~boundary_mask
        valid_points = np.sum(valid_area)
        valid_covered = np.sum(coverage_grid & valid_area)
        
        # Calculate fitness with rewards for pit coverage and penalties for avoidance coverage
        if pit_points > 0:
            pit_ratio = pit_covered / pit_points
        else:
            pit_ratio = 0
            
        if avoidance_points > 0:
            avoidance_ratio = avoidance_covered / avoidance_points
        else:
            avoidance_ratio = 0
            
        if valid_points > 0:
            total_ratio = valid_covered / valid_points
        else:
            total_ratio = 0
        
        # Fitness function with strong emphasis on pit coverage and penalty for avoidance coverage
        fitness = (0.7 * pit_ratio) + (0.3 * total_ratio) - (0.5 * avoidance_ratio)
        
        # Check if any station is in a pit (should be avoided)
        for x, y in stations:
            if any(is_inside_polygon(x, y, pit) for pit in self.pits):
                fitness -= 0.5  # Heavy penalty
        
        return max(0.0, fitness)  # Ensure non-negative fitness

    def update_roles(self):
        """Update producer and scrounger roles based on fitness"""
        sorted_indices = np.argsort(-self.fitness)
        num_producers = max(1, int(0.2 * self.num_sparrows))
        self.producers = sorted_indices[:num_producers]
        self.scroungers = sorted_indices[num_producers:]
    
    def update_producers(self, iteration):
        """Update producer positions with adaptive step size"""
        step_size = 0.2 * (1 - iteration / self.max_iter)
        for i in self.producers:
            for j in range(len(self.positions[i])):
                # Update with random movement
                self.positions[i][j] += np.random.uniform(-step_size, step_size, 2)
                
                # Ensure within bounds
                self.positions[i][j] = np.clip(self.positions[i][j], 0, self.area_size[0])
                
                # Ensure not in pit areas
                attempts = 0
                while any(is_inside_polygon(self.positions[i][j][0], self.positions[i][j][1], pit) for pit in self.pits) and attempts < 5:
                    self.positions[i][j] += np.random.uniform(-step_size, step_size, 2)
                    self.positions[i][j] = np.clip(self.positions[i][j], 0, self.area_size[0])
                    attempts += 1
                
                # If still in pit after attempts, place randomly outside pits
                if any(is_inside_polygon(self.positions[i][j][0], self.positions[i][j][1], pit) for pit in self.pits):
                    valid = False
                    while not valid:
                        self.positions[i][j] = np.random.uniform(0, self.area_size[0], 2)
                        valid = not any(is_inside_polygon(self.positions[i][j][0], self.positions[i][j][1], pit) for pit in self.pits)
    
    def update_scroungers(self):
        """Update scrounger positions by following producers"""
        for i in self.scroungers:
            # Choose a random producer to follow
            producer_idx = np.random.choice(self.producers)
            producer_pos = self.positions[producer_idx]
            
            for j in range(len(self.positions[i])):
                # Move towards producer with some randomness
                A = np.random.uniform(-0.5, 1.5, 2)  # Random coefficient
                step = A * (producer_pos[j] - self.positions[i][j])
                self.positions[i][j] += step
                
                # Ensure within bounds
                self.positions[i][j] = np.clip(self.positions[i][j], 0, self.area_size[0])
                
                # Ensure not in pit areas
                if any(is_inside_polygon(self.positions[i][j][0], self.positions[i][j][1], pit) for pit in self.pits):
                    valid = False
                    while not valid:
                        self.positions[i][j] = np.random.uniform(0, self.area_size[0], 2)
                        valid = not any(is_inside_polygon(self.positions[i][j][0], self.positions[i][j][1], pit) for pit in self.pits)
    
    def random_walk(self, iteration):
        """Perform Levy flight random walk for exploration"""
        if np.random.rand() < 0.2:  # Only do this 20% of the time
            alpha = 1.5  # Levy exponent
            sigma = (np.math.gamma(1 + alpha) * np.sin(np.pi * alpha / 2) / 
                    (np.math.gamma((1 + alpha) / 2) * alpha * 2**((alpha - 1) / 2)))**(1 / alpha)
            
            # Indices of sparrows to apply random walk (20%)
            walk_indices = np.random.choice(
                self.num_sparrows, 
                size=max(1, int(0.2 * self.num_sparrows)), 
                replace=False
            )
            
            for i in walk_indices:
                u = np.random.normal(0, sigma, size=(self.num_stations, 2))
                v = np.random.normal(0, 1, size=(self.num_stations, 2))
                step_size = u / np.abs(v)**(1 / alpha) * 0.1 * (1 - iteration / self.max_iter)
                
                self.positions[i] += step_size
                self.positions[i] = np.clip(self.positions[i], 0, self.area_size[0])
                
                # Quick check to fix stations in restricted area (pits)
                for j in range(self.num_stations):
                    if any(is_inside_polygon(self.positions[i][j][0], self.positions[i][j][1], pit) for pit in self.pits):
                        valid = False
                        while not valid:
                            self.positions[i][j] = np.random.uniform(0, self.area_size[0], 2)
                            valid = not any(is_inside_polygon(self.positions[i][j][0], self.positions[i][j][1], pit) for pit in self.pits)
    
    def anti_predator_escape(self):
        """Enhance diversity by allowing some sparrows to escape"""
        worst_index = np.argmin(self.fitness)
        best_index = np.argmax(self.fitness)
        
        # Indices of sparrows to apply escape mechanism (10%)
        escape_indices = np.random.choice(
            self.num_sparrows, 
            size=max(1, int(0.1 * self.num_sparrows)), 
            replace=False
        )
        
        for i in escape_indices:
            for j in range(self.num_stations):
                if np.random.rand() < 0.5:
                    # Move away from worst position
                    self.positions[i][j] += np.random.normal(0, 0.5, 2) * (
                        self.positions[best_index][j] - self.positions[worst_index][j]
                    )
                else:
                    # Random position
                    self.positions[i][j] = np.random.uniform(0, self.area_size[0], 2)
                
                # Ensure within bounds
                self.positions[i][j] = np.clip(self.positions[i][j], 0, self.area_size[0])
                
                # Ensure not in restricted area (pits)
                if any(is_inside_polygon(self.positions[i][j][0], self.positions[i][j][1], pit) for pit in self.pits):
                    valid = False
                    while not valid:
                        self.positions[i][j] = np.random.uniform(0, self.area_size[0], 2)
                        valid = not any(is_inside_polygon(self.positions[i][j][0], self.positions[i][j][1], pit) for pit in self.pits)
    
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
            # Update masks for new resolution
            self.current_pit_masks = [create_polygon_mask(self.current_X, self.current_Y, pit) for pit in self.pits]
            self.current_combined_pit_mask = np.any(self.current_pit_masks, axis=0)
            
            self.current_avoidance_masks = [create_polygon_mask(self.current_X, self.current_Y, region) for region in self.avoidance_regions]
            self.current_combined_avoidance_mask = np.any(self.current_avoidance_masks, axis=0)
            
            self.current_boundary_mask = create_boundary_mask(self.current_X, self.current_Y, self.area_size, self.boundary_buffer)
            self.current_total_restricted_mask = self.current_combined_pit_mask | self.current_combined_avoidance_mask | self.current_boundary_mask
    
    def optimize(self):
        """Main optimization loop with early stopping and adaptive resolution"""
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
                self.fitness[i] = self.coverage_ratio_multi_region(self.positions[i])
            
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
        final_fitness = self.coverage_ratio_multi_region(self.best_pos, use_current_grid=False)
        print(f"Final fitness: {final_fitness:.4f}")
        
        total_time = time.time() - start_time
        print(f"Total optimization time: {total_time:.2f} seconds")
        
        return self.best_pos, final_fitness

    def visualize_coverage(self):
        """Visualize the coverage with multiple pits and avoidance regions"""
        L1, L2 = self.area_size
        
        # Reuse final grid for visualization
        X, Y = self.final_X, self.final_Y
        signal_grid = np.zeros_like(X, dtype=float)
    
        # Calculate signal strength for each station
        for x, y in self.best_pos:
            signal_grid += signal_strength_vectorized(X, Y, x, y, self.Rs)
        
        plt.figure(figsize=(12, 9))
        
        # Plot signal strength
        contour = plt.contourf(X, Y, signal_grid, levels=20, cmap='viridis', alpha=0.7)
        cbar = plt.colorbar(contour, label='Signal Strength')
        
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
        
        # Plot all avoidance areas with shading
        for i, avoidance_region in enumerate(self.avoidance_regions):
            plt.fill(avoidance_region[:, 0], avoidance_region[:, 1], 
                    color='gray', alpha=0.2)
            plt.plot(avoidance_region[:, 0], avoidance_region[:, 1], 
                    color='gray', linestyle="--", alpha=0.8)
        
        # Plot all pit areas (target coverage)
        for i, pit in enumerate(self.pits):
            plt.fill(pit[:, 0], pit[:, 1], 
                    color='black', alpha=0.7)
            
        # Add a legend item for pits and avoidance regions
        pit_patch = patches.Patch(color='black', alpha=0.7, label="Pit Areas (Target Coverage)")
        avoid_patch = patches.Patch(color='gray', alpha=0.2, label="Avoidance Areas")
        plt.legend(handles=[pit_patch, avoid_patch], loc='upper right')
        
        # Plot base stations
        plt.scatter(self.best_pos[:, 0], self.best_pos[:, 1], 
                    color='red', marker='x', s=150, linewidth=3, label='Base Stations')
        
        # Draw coverage radius
        for x, y in self.best_pos:
            circle = plt.Circle((x, y), self.Rs, color='blue', fill=False, 
                               linestyle='--', alpha=0.5)
            plt.gca().add_patch(circle)
        
        # Add labels and title
        plt.title("Base Station Coverage with Multiple Target Pit Areas", fontsize=16)
        plt.xlabel("X Coordinate", fontsize=12)
        plt.ylabel("Y Coordinate", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Plot convergence curve
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, len(self.convergence_curve) + 1), self.convergence_curve, 'b-', linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Fitness Value', fontsize=12)
        plt.title('Convergence Curve', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Calculate and display coverage metrics
        self.calculate_coverage_metrics()
    
    def calculate_coverage_metrics(self):
        """Calculate and display coverage metrics for multiple regions"""
        # Reuse final grid for metrics
        X, Y = self.final_X, self.final_Y
        
        # Total area coverage
        coverage_grid = np.zeros_like(X, dtype=bool)
        for x, y in self.best_pos:
            coverage_grid |= is_covered_vectorized(X, Y, x, y, self.Rs)
        
        # Calculate pit area coverage (target area)
        pit_points = np.sum(self.final_combined_pit_mask)
        pit_covered = np.sum(coverage_grid & self.final_combined_pit_mask)
        
        # Calculate avoidance area coverage
        avoidance_points = np.sum(self.final_combined_avoidance_mask & ~self.final_combined_pit_mask)
        avoidance_covered = np.sum(coverage_grid & self.final_combined_avoidance_mask & ~self.final_combined_pit_mask)
        
        # Calculate normal area coverage (outside both pit and avoidance)
        normal_mask = ~(self.final_combined_pit_mask | self.final_combined_avoidance_mask)
        normal_points = np.sum(normal_mask)
        normal_covered = np.sum(coverage_grid & normal_mask)
        
        total_points = X.size
        total_covered = np.sum(coverage_grid)
        
        print("\nCoverage Metrics:")
        print(f"Total Area Coverage: {total_covered / total_points * 100:.2f}%")
        if pit_points > 0:
            print(f"All Pit Areas Coverage (TARGET): {pit_covered / pit_points * 100:.2f}%")
        if avoidance_points > 0:
            print(f"All Avoidance Areas Coverage: {avoidance_covered / avoidance_points * 100:.2f}%")
        if normal_points > 0:
            print(f"Normal Area Coverage: {normal_covered / normal_points * 100:.2f}%")
        
        # Check for stations in restricted areas
        stations_in_pits = 0
        stations_in_avoidance = 0
        
        for x, y in self.best_pos:
            if any(is_inside_polygon(x, y, pit) for pit in self.pits):
                stations_in_pits += 1
            elif any(is_inside_polygon(x, y, region) for region in self.avoidance_regions):
                stations_in_avoidance += 1
        
        print(f"\nStation Placement:")
        print(f"Stations in Pit Areas: {stations_in_pits} (Should be 0)")
        print(f"Stations in Avoidance Areas: {stations_in_avoidance}")
        
        # Calculate overlap between stations
        overlap_count = 0
        for i in range(len(self.best_pos)):
            for j in range(i+1, len(self.best_pos)):
                dist = np.sqrt(np.sum((self.best_pos[i] - self.best_pos[j])**2))
                if dist < 2 * self.Rs:
                    overlap_count += 1
        
        print(f"Station Overlap Count: {overlap_count}")

# Run the algorithm with optimized parameters
if __name__ == "__main__":
    print("Running Optimized SSA for Base Station Placement...")
    
    # Define parameters - reduced for faster execution with similar results
    num_sparrows = 50  # Slightly reduced from 60
    num_stations =  8 # Number of base stations to place  
    area_size = (20 , 20)  # Area size (L1, L2)
    grid_size = (50, 50)  
    Rs = 3.2  
    max_iter = 200  
    early_stop = 50  
    
    # Create and run the algorithm
    ssa = SparrowSearchAlgorithm(
        num_sparrows=num_sparrows,
        num_stations=num_stations,
        area_size=area_size,
        grid_size=grid_size,
        Rs=Rs,
        max_iter=max_iter,
        seed=42,
        early_stop_iter=early_stop
    )
    
    best_solution, best_value = ssa.optimize()
    print("\nOptimization Complete!")
    print("Best Base Station Positions:")
    for i, (x, y) in enumerate(best_solution):
        print(f"Station {i+1}: ({x:.4f}, {y:.4f})")
    print(f"Maximum Coverage Ratio: {best_value:.4f}")
    
    # Visualize results
    ssa.visualize_coverage()