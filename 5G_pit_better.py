import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.spatial import ConvexHull, distance
from numba import njit

# Global variables for shapes
pit_shape = None
avoidance_shape = None

# Three-part perception probability model functions from first code
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

# JIT-compiled helper functions from second code
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

# Polygon related functions from second code
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

def create_polygon_mask(X, Y, polygon):
    """Create a mask for points inside the polygon"""
    mask = np.zeros_like(X, dtype=bool)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            mask[i, j] = is_inside_polygon(X[i, j], Y[i, j], polygon)
    return mask

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

# Enhanced coverage ratio that integrates both approaches
def enhanced_coverage_ratio(base_stations, area_size, grid_size, Rs, Re=1.0, lambda_param=1.0, 
                            pit_mask=None, avoidance_mask=None, boundary_mask=None, X=None, Y=None, 
                            weight_pit=3.0, use_probability_model=True):
    """
    Enhanced coverage ratio calculation that can handle both probability-based and 
    constraint-based approaches, with optional pit/avoidance area priorities
    """
    global pit_shape, avoidance_shape
    
    L1, L2 = area_size
    m, n = grid_size
    
    # If grid points not provided, create them
    if X is None or Y is None:
        x_grid = np.linspace(0, L1, m)
        y_grid = np.linspace(0, L2, n)
        X, Y = np.meshgrid(x_grid, y_grid)
    
    # Calculate coverage based on selected approach
    if use_probability_model:
        # Use probability model from first code
        prob_grid = joint_probability(base_stations, X, Y, Rs, Re, lambda_param)
        coverage_grid = prob_grid > 0.5  # For binary coverage analysis
    else:
        # Use simple coverage model (point is covered or not)
        coverage_grid = np.zeros_like(X, dtype=bool)
        for x, y in base_stations:
            coverage_grid |= is_covered_vectorized(X, Y, x, y, Rs)
    
    # Calculate basic coverage statistics
    total_points = X.size
    total_covered = np.sum(coverage_grid)
    base_coverage = total_covered / total_points
    
    # If we're using constraint-based approach with pit areas
    if pit_mask is not None and avoidance_mask is not None and boundary_mask is not None:
        # Calculate pit coverage
        pit_points = np.sum(pit_mask)
        pit_covered = np.sum(coverage_grid & pit_mask) if pit_points > 0 else 0
        pit_coverage_ratio = pit_covered / pit_points if pit_points > 0 else 0
        
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
        
        # Calculate clustering penalty
        clustering_penalty = calculate_clustering_penalty(base_stations, Rs, threshold=1.2)
        
        # Combined fitness with pit priority
        return (0.3 * base_coverage + 0.7 * pit_coverage_ratio) - (0.5 * placement_penalty) - (0.15 * clustering_penalty)
        
    else:
        # Original approach from first code - just return coverage ratio
        if use_probability_model:
            return np.mean(prob_grid)  # Average probability across grid
        else:
            return base_coverage

class EnhancedSparrowSearchAlgorithm:
    def __init__(self, num_sparrows=30, num_stations=5, area_size=(10, 10), grid_size=(50, 50), 
                 Rs=2.2, Re=1.0, lambda_param=1.0, max_iter=100, seed=42, 
                 use_constraints=True, use_probability_model=True, early_stop_iter=20, 
                 boundary_buffer=2.0):
        """
        Enhanced Sparrow Search Algorithm that combines probability-based coverage model
        with constraint handling and adaptive grid resolution
        
        Parameters:
        -----------
        num_sparrows : int
            Number of sparrows (solutions) in the population
        num_stations : int
            Number of base stations to optimize
        area_size : tuple
            Size of the area (L1, L2)
        grid_size : tuple
            Resolution of the grid (m, n)
        Rs : float
            Coverage radius
        Re : float
            Inner radius for probability model
        lambda_param : float
            Parameter for probability exponential decay
        max_iter : int
            Maximum number of iterations
        seed : int
            Random seed for reproducibility
        use_constraints : bool
            Whether to use constraint-based approach with pit/avoidance areas
        use_probability_model : bool
            Whether to use probability-based coverage model or simple coverage model
        early_stop_iter : int
            Number of iterations without improvement to trigger early stopping
        boundary_buffer : float
            Buffer distance from boundaries
        """
        global pit_shape, avoidance_shape
        
        # Basic parameters
        self.num_sparrows = num_sparrows
        self.num_stations = num_stations
        self.area_size = area_size
        self.grid_size = grid_size
        self.Rs = Rs
        self.Re = Re
        self.lambda_param = lambda_param
        self.max_iter = max_iter
        self.use_constraints = use_constraints
        self.use_probability_model = use_probability_model
        self.early_stop_iter = early_stop_iter
        self.boundary_buffer = boundary_buffer
        
        # Set random seed
        np.random.seed(seed)
        
        # Set up pit and avoidance areas if using constraints
        if use_constraints:
            self.pit_center = (area_size[0] / 2, area_size[1] / 2)
            self.pit_shape = generate_irregular_shape(self.pit_center, num_points=8, scale=1.5, seed=seed)
            self.avoidance_shape = generate_irregular_shape(self.pit_center, num_points=12, scale=2.5, seed=seed)
            
            # Assign to global variables
            pit_shape = self.pit_shape
            avoidance_shape = self.avoidance_shape
        else:
            self.pit_shape = None
            self.avoidance_shape = None
            pit_shape = None
            avoidance_shape = None
        
        # Pre-compute grid and masks
        L1, L2 = self.area_size
        m, n = self.grid_size
        self.x_grid = np.linspace(0, L1, m)
        self.y_grid = np.linspace(0, L2, n)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)
        
        # Create masks if using constraints
        if use_constraints:
            self.pit_mask = create_polygon_mask(self.X, self.Y, self.pit_shape)
            self.avoidance_mask = create_polygon_mask(self.X, self.Y, self.avoidance_shape)
            self.boundary_mask = create_boundary_mask(self.X, self.Y, self.area_size, self.boundary_buffer)
            self.combined_avoidance_mask = self.avoidance_mask | self.boundary_mask
        else:
            self.pit_mask = None
            self.avoidance_mask = None
            self.boundary_mask = None
            self.combined_avoidance_mask = None
        
        # Initialize positions
        self.positions = self.initialize_positions()
        
        # Adaptive grid resolution - start with lower resolution for faster initial convergence
        self.current_grid_size = (min(25, grid_size[0]), min(25, grid_size[1]))
        self.current_X, self.current_Y = np.meshgrid(
            np.linspace(0, L1, self.current_grid_size[0]),
            np.linspace(0, L2, self.current_grid_size[1])
        )
        
        # Create current masks if using constraints
        if use_constraints:
            self.current_pit_mask = create_polygon_mask(self.current_X, self.current_Y, self.pit_shape)
            self.current_avoidance_mask = create_polygon_mask(self.current_X, self.current_Y, self.avoidance_shape)
            self.current_boundary_mask = create_boundary_mask(self.current_X, self.current_Y, self.area_size, self.boundary_buffer)
            self.current_combined_avoidance_mask = self.current_avoidance_mask | self.current_boundary_mask
        else:
            self.current_pit_mask = None
            self.current_avoidance_mask = None
            self.current_boundary_mask = None
            self.current_combined_avoidance_mask = None
        
        # Calculate initial fitness
        self.fitness = np.array([
            enhanced_coverage_ratio(
                pos, area_size, self.current_grid_size, Rs, Re, lambda_param,
                self.current_pit_mask, self.current_avoidance_mask, self.current_boundary_mask,
                self.current_X, self.current_Y, use_probability_model=use_probability_model
            ) for pos in self.positions
        ])
        
        # Best solution tracking
        self.best_pos = self.positions[np.argmax(self.fitness)]
        self.best_fit = np.max(self.fitness)
        self.history = []
        
        # Convergence tracking
        self.convergence_curve = []
        self.no_improvement_count = 0
        
        # Create validation grid (higher resolution) for final evaluation
        self.final_grid_size = grid_size
        self.final_X, self.final_Y = self.X, self.Y
        if use_constraints:
            self.final_pit_mask = self.pit_mask
            self.final_avoidance_mask = self.avoidance_mask
            self.final_boundary_mask = self.boundary_mask
            self.final_combined_avoidance_mask = self.combined_avoidance_mask
        else:
            self.final_pit_mask = None
            self.final_avoidance_mask = None
            self.final_boundary_mask = None
            self.final_combined_avoidance_mask = None

    def initialize_positions(self):
        """Initialize valid sparrow positions based on problem constraints"""
        if self.use_constraints:
            positions = []
            for _ in range(self.num_sparrows):
                valid_positions = []
                
                # Try to place stations outside all avoidance areas (including boundary buffer)
                while len(valid_positions) < min(3, self.num_stations):
                    # Select positions in the "safe" zone (not in pit, not in avoidance, not in boundary buffer)
                    x = np.random.uniform(self.boundary_buffer, self.area_size[0] - self.boundary_buffer)
                    y = np.random.uniform(self.boundary_buffer, self.area_size[1] - self.boundary_buffer)
                    
                    # Ensure the point is not in any restricted area
                    if (not is_inside_polygon(x, y, self.pit_shape) and 
                        not is_inside_polygon(x, y, self.avoidance_shape) and 
                        not is_in_boundary_buffer(x, y, self.area_size, self.boundary_buffer)):
                        
                        if not valid_positions or all(np.sqrt((x - vx)**2 + (y - vy)**2) > 0.7 * self.Rs 
                              for vx, vy in valid_positions):
                            valid_positions.append([x, y])
                    
                    # If we've tried too many times, just pick a random point outside the pit
                    if len(valid_positions) < 1 and len(valid_positions) < self.num_stations:
                        x, y = np.random.uniform(0, self.area_size[0], 2)
                        if not is_inside_polygon(x, y, self.pit_shape):
                            valid_positions.append([x, y])
                
                # Place remaining stations with more relaxed constraints (outside pit, can be in avoidance or buffer)
                while len(valid_positions) < self.num_stations:
                    x, y = np.random.uniform(0, self.area_size[0], 2)
                    
                    if not is_inside_polygon(x, y, self.pit_shape):
                        valid_positions.append([x, y])
                
                positions.append(valid_positions)
            
            return np.array(positions)
        else:
            # Original initialization from first code (without constraints)
            valid_positions = np.zeros((self.num_sparrows, self.num_stations, 2))
            min_dist = 1.5  # Minimum distance between stations
            min_boundary_dist = 3  # Minimum distance from boundaries
            L1, L2 = self.area_size
            
            for i in range(self.num_sparrows):
                stations = []
                while len(stations) < self.num_stations:
                    x, y = np.random.uniform(min_boundary_dist, L1 - min_boundary_dist), np.random.uniform(min_boundary_dist, L2 - min_boundary_dist)
                    if all(np.linalg.norm(np.array([x, y]) - np.array(s)) >= min_dist for s in stations):
                        stations.append((x, y))
                valid_positions[i] = np.array(stations)
            return valid_positions

    def enforce_constraints(self, position):
        """Ensure a single base station stays within bounds and maintains minimum distance
        
        Parameters:
        position: A single position [x, y] that needs to be constrained
        
        Returns:
        constrained_position: The position after applying constraints
        """
        min_boundary_dist = 3  # Minimum distance from boundaries
        L1, L2 = self.area_size
        
        # First clip to boundary
        constrained_position = np.clip(position, min_boundary_dist, [L1 - min_boundary_dist, L2 - min_boundary_dist])
        
        # Ensure not in pit if using constraints
        if self.use_constraints:
            if is_inside_polygon(constrained_position[0], constrained_position[1], self.pit_shape):
                # Find a new position outside pit
                attempts = 0
                while attempts < 10 and is_inside_polygon(constrained_position[0], constrained_position[1], self.pit_shape):
                    constrained_position = np.random.uniform(min_boundary_dist, L1 - min_boundary_dist, 2)
                    attempts += 1
        
        return constrained_position

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
        
        for i in self.producers:
            curr_iter = iteration + 1
            step_size = 0.2 * (1 - curr_iter / self.max_iter)
            
            for j in range(self.num_stations):
                if np.random.rand() < ST:
                    # Cautious movement
                    exp_factor = np.exp(-i / (alpha * self.max_iter))
                    self.positions[i][j] = self.positions[i][j] * exp_factor
                else:
                    # Explore new areas
                    L = np.random.normal(0, 1, 2)  # Random direction
                    self.positions[i][j] = self.positions[i][j] + step_size * L
                
                # Apply random walk for some producers (20% chance)
                if np.random.rand() < 0.2:
                    self.apply_random_walk_to_station(i, j)
                
                # Enforce constraints
                self.positions[i][j] = self.enforce_constraints(self.positions[i][j])
    
    def apply_random_walk_to_station(self, index, station_index):
        """Apply random walk to a specific station"""
        # Generate random binary sequence
        steps = 20
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
        
        # Get a random point from the walk
        idx = np.random.randint(0, steps)
        
        # Apply normalized position using equation 12
        pos_x = self.positions[index][station_index][0]
        pos_y = self.positions[index][station_index][1]
        
        # Normalize and apply the walk
        new_x = random_walk_normalization(walk_x[idx], min_x, max_x, 0, self.area_size[0])
        new_y = random_walk_normalization(walk_y[idx], min_y, max_y, 0, self.area_size[1])
        
        # Combine current position with random walk influence
        self.positions[index][station_index][0] = 0.7 * pos_x + 0.3 * new_x
        self.positions[index][station_index][1] = 0.7 * pos_y + 0.3 * new_y
    
    def update_scroungers(self):
        """Update scrounger positions by following producers"""
        best_producer_idx = self.producers[0]  # Get best producer
        
        for i in self.scroungers:
            if np.random.rand() > 0.8:  # 20% chance for alert behavior
                # Move away from worst position (anti-predator)
                worst_idx = np.argmin(self.fitness)
                K = np.random.uniform(-1, 1)
                for j in range(self.num_stations):
                    self.positions[i][j] += K * np.abs(self.positions[i][j] - self.positions[worst_idx][j])
            else:
                # Normal joiner behavior - follow best producer
                for j in range(self.num_stations):
                    A = np.random.randn(2)
                    self.positions[i][j] += np.random.rand() * A * (self.positions[best_producer_idx][j] - self.positions[i][j])
            
            # Enforce constraints for each station
            for j in range(self.num_stations):
                self.positions[i][j] = self.enforce_constraints(self.positions[i][j])
    
    def random_walk(self, iteration):
        """Perform Levy flight random walk for exploration"""
        if np.random.rand() < 0.15:  # Only do this 20% of the time
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
                for j in range(self.num_stations):
                    u = np.random.normal(0, sigma, 2)
                    v = np.random.normal(0, 1, 2)
                    step_size = u / np.abs(v)**(1 / alpha) * 0.1 * (1 - iteration / self.max_iter)
                    
                    self.positions[i][j] += step_size
                    self.positions[i][j] = self.enforce_constraints(self.positions[i][j])
    
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
                
                # Ensure valid position
                self.positions[i][j] = self.enforce_constraints(self.positions[i][j])
    
    def adjust_grid_resolution(self, iteration):
        """Dynamically adjust grid resolution based on iteration"""
        if iteration > self.max_iter // 2 and iteration % 10 == 0:
            # Switch to higher resolution for fine-tuning
            self.current_grid_size = (
                min(self.current_grid_size[0] * 1.5, self.final_grid_size[0]),
                min(self.current_grid_size[1] * 1.5, self.final_grid_size[1])
            )
            self.current_grid_size = (int(self.current_grid_size[0]), int(self.current_grid_size[1]))
            
            L1, L2 = self.area_size
            self.current_X, self.current_Y = np.meshgrid(
                np.linspace(0, L1, self.current_grid_size[0]),
                np.linspace(0, L2, self.current_grid_size[1])
            )
            
            if self.use_constraints:
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
            
            # Batch validation to avoid repeated for loops
            valid_solutions = np.ones(self.num_sparrows, dtype=bool)
            
            # Calculate fitness for all solutions
            for i in range(self.num_sparrows):
                self.fitness[i] = enhanced_coverage_ratio(
                    self.positions[i], self.area_size, self.current_grid_size, self.Rs, 
                    self.Re, self.lambda_param,
                    self.current_pit_mask, self.current_avoidance_mask,
                    self.current_boundary_mask, self.current_X, self.current_Y,
                    use_probability_model=self.use_probability_model
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
        final_fitness = enhanced_coverage_ratio(
            self.best_pos, self.area_size, self.final_grid_size, self.Rs,
            self.Re, self.lambda_param,
            self.final_pit_mask, self.final_avoidance_mask,
            self.final_boundary_mask, self.final_X, self.final_Y,
            use_probability_model=self.use_probability_model
        )
        print(f"Final fitness: {final_fitness:.4f}")
        
        total_time = time.time() - start_time
        print(f"Total optimization time: {total_time:.2f} seconds")
        
        return self.best_pos, final_fitness

    def visualize_coverage(self):
        """Visualize the coverage with improved preferences"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        L1, L2 = self.area_size
        m, n = self.final_grid_size
        
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
        
        # Plot avoidance area with shading if using constraints
        if self.use_constraints:
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
        """Calculate and display coverage metrics"""
        # Reuse final grid for metrics
        X, Y = self.final_X, self.final_Y
        
        # Total area coverage
        coverage_grid = np.zeros_like(X, dtype=bool)
        for x, y in self.best_pos:
            coverage_grid |= is_covered_vectorized(X, Y, x, y, self.Rs)
        
        # Calculate pit area coverage (target area) if using constraints
        if self.use_constraints:
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
        
        if self.use_constraints:
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
        
if __name__ == "__main__":
    print("Running Optimized SSA for Base Station Placement...")
    
    # Define parameters - reduced for faster execution with similar results
    num_sparrows = 50  # Slightly reduced from 60
    num_stations = 5  # Number of base stations to place  
    area_size = (15,15)  # Area size (L1, L2)
    grid_size = (100, 100)  
    Rs = 2.5  # Coverage radius (Rs) 
    max_iter = 100  
    early_stop = 50  
    
    # Create and run the algorithm
    ssa = EnhancedSparrowSearchAlgorithm(
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