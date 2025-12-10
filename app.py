import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, distance
import matplotlib.patches as patches
import time  # For timing

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

def calculate_clustering_penalty(base_stations, Rs, threshold=0.5):
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
    penalty_factors = np.exp(3 * (threshold * Rs - close_distances)) - 1


    return np.sum(penalty_factors) / (len(base_stations) * (len(base_stations) - 1) / 2)

def coverage_ratio_multi_region(base_stations, area_size, grid_size, Rs, Re=1.0, lambda_param=1.0,
                    pit_masks=None, avoidance_masks=None, boundary_mask=None,
                    X=None, Y=None, pit_centers=None, pits=None, avoidance_regions=None, weight_pit=3.0,
                    cost_weight=1.5, auto_suggest=False):
    """Calculate coverage ratio with multiple pit and avoidance regions"""

    import numpy as np

    L1, L2 = area_size
    m, n = grid_size

    # Convert base_stations to numpy array for vectorized operations
    base_stations_array = np.array(base_stations)

    # --- Helper: Suggest candidate locations based on low probability ---
    def find_low_coverage_candidates(prob_grid, boundary_mask, threshold=0.2, top_k=3):
        low_cov = (prob_grid < threshold) & (boundary_mask if boundary_mask is not None else True)
        candidate_coords = np.argwhere(low_cov)
        scores = prob_grid[low_cov]
        sorted_indices = np.argsort(scores)[:top_k]
        return [tuple(map(float, [X[i, j], Y[i, j]])) for i, j in candidate_coords[sorted_indices]]

    # Step 1: Evaluate initial probability grid
    prob_grid = joint_probability_vectorized(X, Y, base_stations_array, Rs, Re, lambda_param)

    # Step 2 (Optional): Auto-suggest new station for better coverage
    if auto_suggest:
        candidates = find_low_coverage_candidates(prob_grid, boundary_mask, threshold=0.25, top_k=1)
        for cand in candidates:
            base_stations.append(cand)
        base_stations_array = np.array(base_stations)
        prob_grid = joint_probability_vectorized(X, Y, base_stations_array, Rs, Re, lambda_param)

    # Step 3: Total coverage (inner region)
    if boundary_mask is not None:
        inner_region_points = np.sum(boundary_mask)
        total_coverage = np.sum(prob_grid[boundary_mask]) / inner_region_points if inner_region_points > 0 else 0
    else:
        total_points = X.size
        total_coverage = np.sum(prob_grid) / total_points

    # Step 4: Pit coverage
    if pit_masks is not None and len(pit_masks) > 0:
        pit_coverage_ratios = []
        for pit_mask in pit_masks:
            pit_points = np.sum(pit_mask)
            if pit_points > 0:
                pit_covered = np.sum(prob_grid[pit_mask]) / pit_points
                pit_coverage_ratios.append(pit_covered)
        pit_coverage = np.mean(pit_coverage_ratios) if pit_coverage_ratios else 0
    else:
        pit_coverage = 0

    # Step 5: Penalties for placement
    placement_penalty = 0
    for x, y in base_stations:
        in_pit = any(is_inside_polygon(x, y, pit) for pit in pits)
        if in_pit:
            placement_penalty += 10.0
            continue
        in_avoidance = any(is_inside_polygon(x, y, region) for region in avoidance_regions)
        if in_avoidance:
            placement_penalty += 0.2
        elif is_in_boundary_buffer(x, y, area_size):
            placement_penalty += 200.0

    if len(base_stations) > 0:
        placement_penalty /= len(base_stations)

    # Step 6: Clustering penalty
    clustering_penalty = calculate_clustering_penalty(base_stations, Rs, threshold=0.5)

    # Step 7: Radius and cost penalties
    characteristic_length = np.sqrt(area_size[0] * area_size[1])
    relative_radius = Rs / characteristic_length
    radius_threshold = 0.05

    base_radius_penalty = 1.2 * (Rs ** 3) / characteristic_length
    additional_penalty = 10.0 * (relative_radius - radius_threshold) ** 2 if relative_radius > radius_threshold else 0
    total_area = area_size[0] * area_size[1]
    theoretical_max_coverage = min(1.0, len(base_stations) * np.pi * Rs**2 / total_area)
    efficiency_score = total_coverage / (theoretical_max_coverage + 1e-6)
    radius_penalty = 0.5 * base_radius_penalty + additional_penalty * (1.0 - efficiency_score)

    base_cost = len(base_stations) * 1.0
    radius_cost = len(base_stations) * Rs ** 1.2
    if Rs > 0.12 * characteristic_length:
        premium_cost = len(base_stations) * 3.0 * ((Rs - 0.12 * characteristic_length) ** 3)
        radius_cost += premium_cost

    total_cost = base_cost + radius_cost * np.exp(0.7 * Rs)
    scaled_cost = cost_weight * total_cost / (characteristic_length )

    min_coverage_reward = -30.0 * (0.90 - total_coverage) ** 1.5 if total_coverage < 0.90 else 0
    pit_reward = np.exp(4 * pit_coverage) - 1

    station_efficiency = total_coverage / (len(base_stations) + 1e-6)
    station_reward = 2.0 * np.log(1 + len(base_stations))
    efficiency_reward = 6.0 * station_efficiency

    return (
        30 * total_coverage +
        20.0 * pit_reward +
        station_reward +
        efficiency_reward -
        5.0 * placement_penalty -
        35.0 * clustering_penalty -
        scaled_cost +
        min_coverage_reward
    )




class EnhancedSparrowSearchAlgorithm:
    def __init__(self, num_sparrows=30, num_stations=5, area_size=(10, 10), grid_size=(50, 50), Rs=2, max_iter=100, 
                 seed=42, early_stop_iter=20, boundary_buffer=2.0, num_pits=4, pit_sizes=None):
        
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
        
        # NEW: Allow custom pit sizes
        self.pit_sizes = pit_sizes if pit_sizes is not None else [1.5] * num_pits
        
        # Ensure we have the right number of pit sizes
        if len(self.pit_sizes) < self.num_pits:
            # If fewer sizes than pits, repeat the last size
            self.pit_sizes.extend([self.pit_sizes[-1]] * (self.num_pits - len(self.pit_sizes)))
        elif len(self.pit_sizes) > self.num_pits:
            # If more sizes than pits, truncate
            self.pit_sizes = self.pit_sizes[:self.num_pits]
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Generate pit areas first with individual sizes
        self.pit_centers, self.pits = self.generate_pit_regions(self.num_pits, self.pit_sizes, min_distance=3.0)
        
        # Generate avoidance areas centered on the same locations as pits but with larger scale
        # Use 1.5x the pit size for each avoidance region
        avoidance_sizes = [size * 1.5 for size in self.pit_sizes]
        self.avoidance_regions = self.generate_avoidance_regions(self.pit_centers, avoidance_sizes)
        
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

    def generate_pit_regions(self, num_regions, pit_scales, min_distance):
        """Generate multiple non-overlapping pit regions with individual sizes and return their centers and shapes"""
        regions = []
        centers = []
        attempts = 0
        max_attempts = 100  # Avoid infinite loop
        
        while len(regions) < num_regions and attempts < max_attempts:
            # Get the current pit size
            current_pit_scale = pit_scales[len(regions)]
            
            # Generate random center point
            center = (
                np.random.uniform(self.boundary_buffer + current_pit_scale, self.area_size[0] - self.boundary_buffer - current_pit_scale),
                np.random.uniform(self.boundary_buffer + current_pit_scale, self.area_size[1] - self.boundary_buffer - current_pit_scale)
            )
            
            # Check if this center is far enough from existing regions
            if regions and any(np.sqrt((center[0] - c[0])**2 + (center[1] - c[1])**2) < min_distance for c in centers):
                attempts += 1
                continue
            
            # Generate irregular shape with the specific size for this pit
            new_region = generate_irregular_shape(center, num_points=8, scale=current_pit_scale, seed=None)
            
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

    def generate_avoidance_regions(self, centers, avoidance_scales):
        """Generate avoidance regions around the same centers as pit regions but with individual larger scales"""
        avoidance_regions = []
        
        for i, center in enumerate(centers):
            # Get the specific avoidance scale for this region
            avoidance_scale = avoidance_scales[i]
            
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

    def coverage_ratio_multi_region(self, base_stations):
        """Wrapper for the coverage ratio function with multiple regions"""
        return coverage_ratio_multi_region(
            base_stations, self.area_size, self.current_grid_size, self.Rs,
            pit_masks=self.current_pit_masks, 
            avoidance_masks=self.current_avoidance_masks,
            boundary_mask=self.current_boundary_mask,
            X=self.current_X, Y=self.current_Y,
            pit_centers=self.pit_centers,
            pits=self.pits,
            avoidance_regions=self.avoidance_regions
        )

    def initialize_positions(self):
        """Initialize sparrow positions with stations outside restricted areas"""
        positions = []
        min_dist = self.Rs * 1.5  # Minimum distance between stations
        
        for _ in range(self.num_sparrows):
            valid_positions = []
            
            # Try to place stations outside all avoidance areas (including boundary buffer)
            attempts = 0
            max_attempts = 1000  # Prevent infinite loop
            
            while len(valid_positions) < min(3, self.num_stations) and attempts < max_attempts:
                attempts += 1
                
                # Select positions with boundary_buffer from edges
                x = np.random.uniform(self.boundary_buffer, self.area_size[0] - self.boundary_buffer)
                y = np.random.uniform(self.boundary_buffer, self.area_size[1] - self.boundary_buffer)
                
                # Ensure the point is not in any restricted area
                in_pit = False
                for pit in self.pits:
                    if is_inside_polygon(x, y, pit):
                        in_pit = True
                        break
                
                in_avoidance = False
                if not in_pit:
                    for region in self.avoidance_regions:
                        if is_inside_polygon(x, y, region):
                            in_avoidance = True
                            break
                
                if not in_pit and not in_avoidance and not is_in_boundary_buffer(x, y, self.area_size, self.boundary_buffer):
                    # Apply minimum distance constraint between stations
                    if not valid_positions or all(np.sqrt((x - vx)**2 + (y - vy)**2) >= min_dist 
                        for vx, vy in valid_positions):
                        valid_positions.append([x, y])
                
                # If we've tried too many times, just pick a random point outside all pits
                if attempts >= max_attempts and len(valid_positions) < 1:
                    x = np.random.uniform(self.boundary_buffer, self.area_size[0] - self.boundary_buffer)
                    y = np.random.uniform(self.boundary_buffer, self.area_size[1] - self.boundary_buffer)
                    
                    in_pit = False
                    for pit in self.pits:
                        if is_inside_polygon(x, y, pit):
                            in_pit = True
                            break
                    
                    if not in_pit:
                        valid_positions.append([x, y])
            
            # Place remaining stations with more relaxed constraints (outside pits, can be in avoidance)
            attempts = 0
            while len(valid_positions) < self.num_stations and attempts < max_attempts:
                attempts += 1
                x = np.random.uniform(self.boundary_buffer, self.area_size[0] - self.boundary_buffer)
                y = np.random.uniform(self.boundary_buffer, self.area_size[1] - self.boundary_buffer)
                
                in_pit = False
                for pit in self.pits:
                    if is_inside_polygon(x, y, pit):
                        in_pit = True
                        break
                
                if not in_pit:
                    # Apply minimum distance constraint between stations
                    if all(np.sqrt((x - vx)**2 + (y - vy)**2) >= min_dist for vx, vy in valid_positions):
                        valid_positions.append([x, y])
            
            # If we still don't have enough stations after all attempts, add them anyway with just pit constraint
            while len(valid_positions) < self.num_stations:
                x = np.random.uniform(self.boundary_buffer, self.area_size[0] - self.boundary_buffer)
                y = np.random.uniform(self.boundary_buffer, self.area_size[1] - self.boundary_buffer)
                
                in_pit = False
                for pit in self.pits:
                    if is_inside_polygon(x, y, pit):
                        in_pit = True
                        break
                
                if not in_pit:
                    valid_positions.append([x, y])
            
            positions.append(valid_positions)
        
        return np.array(positions)

    def enforce_constraints(self, position):
        """Ensure a single base station stays within bounds and not in pit areas"""
        # First clip to boundary
        constrained_position = np.clip(position, 0, self.area_size[0])
        
        # Ensure not in any pit
        in_pit = False
        for pit in self.pits:
            if is_inside_polygon(constrained_position[0], constrained_position[1], pit):
                in_pit = True
                break
        
        if in_pit:
            # Find a new position outside all pits
            attempts = 0
            while attempts < 10 and in_pit:
                constrained_position = np.random.uniform(self.boundary_buffer, self.area_size[0] - self.boundary_buffer, 2)
                
                in_pit = False
                for pit in self.pits:
                    if is_inside_polygon(constrained_position[0], constrained_position[1], pit):
                        in_pit = True
                        break
                
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
        if np.random.rand() < 0.15:  # Only do this 15% of the time
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
        final_fitness = coverage_ratio_multi_region(
            self.best_pos, self.area_size, self.final_grid_size, self.Rs,
            pit_masks=self.final_pit_masks, 
            avoidance_masks=self.final_avoidance_masks,
            boundary_mask=self.final_boundary_mask, 
            X=self.final_X, Y=self.final_Y,
            pit_centers=self.pit_centers,
            pits=self.pits,
            avoidance_regions=self.avoidance_regions
        )
        print(f"Final fitness: {final_fitness:.4f}")
        
        total_time = time.time() - start_time
        print(f"Total optimization time: {total_time:.2f} seconds")
        
        return self.best_pos, final_fitness

    def visualize_coverage(self):
        """Visualize the coverage with multiple pit and avoidance regions and annotations"""
        import matplotlib.patches as patches
        from matplotlib.lines import Line2D

        L1, L2 = self.area_size
        X, Y = self.final_X, self.final_Y

        # Compute probability grid
        base_stations_array = np.array(self.best_pos)
        prob_grid = joint_probability_vectorized(X, Y, base_stations_array, self.Rs, Re=self.Rs/2, lambda_param=1.0)

        plt.figure(figsize=(14, 10))

        # Plot joint probability as heatmap
        contour = plt.contourf(X, Y, prob_grid, levels=20, cmap='viridis', alpha=0.7)
        cbar = plt.colorbar(contour, label='Coverage Probability')

        # Plot boundary and inner buffer rectangle
        boundary_rect = patches.Rectangle((0, 0), L1, L2, linewidth=2, edgecolor='red', facecolor='none')
        inner_rect = patches.Rectangle(
            (self.boundary_buffer, self.boundary_buffer),
            L1 - 2 * self.boundary_buffer,
            L2 - 2 * self.boundary_buffer,
            linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
        )
        ax = plt.gca()
        ax.add_patch(boundary_rect)
        ax.add_patch(inner_rect)

        # Shade buffer zone
        plt.fill_between([0, L1], 0, self.boundary_buffer, color='gray', alpha=0.2)
        plt.fill_between([0, L1], L2 - self.boundary_buffer, L2, color='gray', alpha=0.2)
        plt.fill_between([0, self.boundary_buffer], self.boundary_buffer, L2 - self.boundary_buffer, color='gray', alpha=0.2)
        plt.fill_between([L1 - self.boundary_buffer, L1], self.boundary_buffer, L2 - self.boundary_buffer, color='gray', alpha=0.2)

        # Annotate corners of buffer zone
        buffer_coords = [
            (self.boundary_buffer, self.boundary_buffer),
            (L1 - self.boundary_buffer, self.boundary_buffer),
            (self.boundary_buffer, L2 - self.boundary_buffer),
            (L1 - self.boundary_buffer, L2 - self.boundary_buffer)
        ]
        for x, y in buffer_coords:
            plt.text(x, y, f"({x:.1f}, {y:.1f})", fontsize=8, color='black', ha='center', va='center')

        # Plot all avoidance zones
        avoidance_colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(self.avoidance_regions))))
        for i, region in enumerate(self.avoidance_regions):
            plt.fill(region[:, 0], region[:, 1], color=avoidance_colors[i], alpha=0.2)
            plt.plot(region[:, 0], region[:, 1], color=avoidance_colors[i], linestyle="--", alpha=0.8)
            # Label
            cx, cy = np.mean(region[:, 0]), np.mean(region[:, 1])
            

        # Plot all pit areas (target zones)
        pit_colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(self.pits))))
        for i, pit in enumerate(self.pits):
            plt.fill(pit[:, 0], pit[:, 1], color=pit_colors[i], alpha=0.7)
            center_x = np.mean(pit[:, 0])
            center_y = np.mean(pit[:, 1])
            plt.text(center_x, center_y, f"Pit-{i+1}\nSize: {self.pit_sizes[i]:.1f}",
                    ha='center', va='center', fontsize=9, color='white', fontweight='bold')

        # Plot base stations
        plt.scatter(self.best_pos[:, 0], self.best_pos[:, 1], color='red', marker='x', s=150, linewidth=3)

        # Annotate base stations
        for idx, (x, y) in enumerate(self.best_pos):
            plt.text(x, y + 1.5, f"BS-{idx+1}\n({x:.1f}, {y:.1f})", ha='center', fontsize=8, color='red')

        # Plot coverage radius circles
        for x, y in self.best_pos:
            circle = plt.Circle((x, y), self.Rs, color='blue', fill=False, linestyle='--', alpha=0.5)
            ax.add_patch(circle)

        # Titles and formatting
        plt.title("Base Station Coverage Map", fontsize=18, fontweight='bold')
        # plt.suptitle(
        #     f"Total Base Stations: {len(self.best_pos)} | Pit Areas: {len(self.pits)} | Avoidance Zones: {len(self.avoidance_regions)}",
        #     fontsize=12, y=0.93, color='gray'
        # )
        plt.xlabel("X Coordinate", fontsize=12)
        plt.ylabel("Y Coordinate", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()

        # Custom legend elements
        legend_elements = [
            Line2D([0], [0], color='red', lw=2, label='Boundary'),
            Line2D([0], [0], color='red', lw=2, linestyle='--', label='Inner Buffer'),
            Line2D([0], [0], marker='x', color='red', markersize=10, linestyle='None', label='Base Station'),
            Line2D([0], [0], color='blue', linestyle='--', lw=2, label='Coverage Radius')
        ]
        if self.avoidance_regions:
            legend_elements.append(patches.Patch(facecolor=avoidance_colors[0], alpha=0.2, label='Avoidance Zone'))
        if self.pits:
            legend_elements.append(patches.Patch(facecolor=pit_colors[0], alpha=0.7, label='Pit Area'))

        plt.legend(handles=legend_elements, loc='upper right', fontsize=10, frameon=True)
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
        """Calculate and display coverage metrics for multiple pit regions"""
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
        
        # Calculate coverage for each pit area
        print("\nCoverage Metrics:")
        print(f"Total Pit Areas: {len(self.pits)}")
        
        total_pit_points = 0
        total_pit_covered = 0
        
        for i, pit_mask in enumerate(self.final_pit_masks):
            pit_points = np.sum(pit_mask)
            pit_covered = np.sum(coverage_grid & pit_mask)
            total_pit_points += pit_points
            total_pit_covered += pit_covered
            
            print(f"Pit Area {i+1} (Size: {self.pit_sizes[i]:.1f}) Coverage: {pit_covered / pit_points * 100:.2f}%")
        
        # Calculate overall pit coverage
        if total_pit_points > 0:
            print(f"Overall Pit Areas Coverage: {total_pit_covered / total_pit_points * 100:.2f}%")
        
        # Calculate avoidance area coverage
        total_avoidance_points = 0
        total_avoidance_covered = 0
        
        for i, avoidance_mask in enumerate(self.final_avoidance_masks):
            # Exclude pit areas from avoidance areas
            exclusive_avoidance = avoidance_mask & ~self.final_combined_pit_mask
            avoidance_points = np.sum(exclusive_avoidance)
            avoidance_covered = np.sum(coverage_grid & exclusive_avoidance)
            
            total_avoidance_points += avoidance_points
            total_avoidance_covered += avoidance_covered
            
            if avoidance_points > 0:
                print(f"Avoidance Area {i+1} Coverage: {avoidance_covered / avoidance_points * 100:.2f}%")
        
        if total_avoidance_points > 0:
            print(f"Overall Avoidance Areas Coverage: {total_avoidance_covered / total_avoidance_points * 100:.2f}%")
        
        # Calculate normal area coverage (outside both pits and avoidance)
        normal_mask = ~(self.final_combined_pit_mask | self.final_combined_avoidance_mask)
        normal_points = np.sum(normal_mask)
        normal_covered = np.sum(coverage_grid & normal_mask)
        
        if normal_points > 0:
            print(f"Normal Area Coverage: {normal_covered / normal_points * 100:.2f}%")
        
        # Total area coverage
        total_points = X.size
        total_covered = np.sum(coverage_grid)
        print(f"Total Area Coverage: {total_covered / total_points * 100:.2f}%")
        
        # Check for stations in restricted areas
        stations_in_pit = 0
        stations_in_avoidance = 0
        stations_in_boundary = 0
        
        for x, y in self.best_pos:
            # Check if in any pit
            in_pit = False
            for pit in self.pits:
                if is_inside_polygon(x, y, pit):
                    in_pit = True
                    stations_in_pit += 1
                    break
            
            if not in_pit:
                # Check if in any avoidance area
                in_avoidance = False
                for region in self.avoidance_regions:
                    if is_inside_polygon(x, y, region):
                        in_avoidance = True
                        stations_in_avoidance += 1
                        break
                
                # Check if in boundary buffer
                if not in_avoidance and is_in_boundary_buffer(x, y, self.area_size, self.boundary_buffer):
                    stations_in_boundary += 1
        
        print(f"\nStation Placement:")
        print(f"Stations in Pit Areas: {stations_in_pit} (Should be 0)")
        print(f"Stations in Avoidance Areas: {stations_in_avoidance}")
        print(f"Stations in Boundary Buffer: {stations_in_boundary}")
        
        # Calculate overlap between stations
        overlap_count = 0
        for i in range(len(self.best_pos)):
            for j in range(i+1, len(self.best_pos)):
                dist = np.sqrt(np.sum((self.best_pos[i] - self.best_pos[j])**2))
                if dist < 2 * self.Rs:
                    overlap_count += 1
        
        print(f"Station Overlap Count: {overlap_count}")

# Run the algorithm with multiple pit regions of different sizes
if __name__ == "__main__":
    print("Running Base Station Placement with Fixed Parameters...")

    # Fixed parameters
    area_size = (10,10)
    grid_size = (100, 100)
    max_iter = 300
    early_stop = 50
    num_sparrows = 100
    num_pits = 2  # Number of pit regions to generate
    pit_sizes = [1.0 , 1.0]  # Define different sizes for each pit
    
    # Fixed station count and radius instead of parameter sweep
    num_stations = 2  # Fixed number of stations
    rs = 2  # Fixed radius

    print(f"Running optimization with fixed parameters: {num_stations} stations and Rs = {rs}")

    # Run the Enhanced SSA algorithm
    ssa = EnhancedSparrowSearchAlgorithm(
        num_sparrows=num_sparrows,
        num_stations=num_stations,
        area_size=area_size,
        grid_size=grid_size,
        Rs=rs,
        max_iter=max_iter,
        seed=42,
        early_stop_iter=early_stop,
        num_pits=num_pits,
        pit_sizes=pit_sizes
    )

    solution, value = ssa.optimize()

    print("\n===== Optimization Results =====")
    print(f"Parameters: {num_stations} stations with Rs = {rs}")
    print(f"Fitness Value: {value:.4f}")
    print("Base Station Positions:")
    for i, (x, y) in enumerate(solution):
        print(f"Station {i+1}: ({x:.4f}, {y:.4f})")

    # Save results to JSON file
    import json
    with open('optimization_results.json', 'w') as f:
        result_data = {
            'num_stations': num_stations,
            'Rs': rs,
            'fitness': value,
            'solution': solution.tolist()
        }
        json.dump(result_data, f, indent=2)

    print("\nResults saved to optimization_results.json")

    # Visualization
    print("\nGenerating visualization...")
    ssa.visualize_coverage()
