
"""AI-based UAV placement using Particle Swarm Optimization."""

import numpy as np

from simulation import DisasterNetworkSimulation, PriorityZone


OBJECTIVE_WEIGHTS = {
    # Main dissertation setting: user connectivity and priority-zone coverage
    # carry most of the score, because the topic is priority-aware placement.
    "connected_users": 0.40,
    "priority_coverage": 0.35,
    "area_coverage": 0.10,
    "relay_links": 0.10,
    "movement_cost": 0.05,
}


WEIGHT_PROFILES = {
    "Priority-aware main": OBJECTIVE_WEIGHTS,
    "User-focused": {
        "connected_users": 0.60,
        "priority_coverage": 0.15,
        "area_coverage": 0.10,
        "relay_links": 0.10,
        "movement_cost": 0.05,
    },
    "Coverage-focused": {
        "connected_users": 0.35,
        "priority_coverage": 0.25,
        "area_coverage": 0.25,
        "relay_links": 0.10,
        "movement_cost": 0.05,
    },
    "Energy-aware": {
        "connected_users": 0.40,
        "priority_coverage": 0.25,
        "area_coverage": 0.10,
        "relay_links": 0.10,
        "movement_cost": 0.15,
    },
}


class ParticleSwarmOptimizer:
    """Search for better UAV x/y positions using PSO."""

    def __init__(
        self,
        simulation: DisasterNetworkSimulation,
        users: np.ndarray,
        initial_drones: np.ndarray,
        priority_zones: list[PriorityZone] | None = None,
        swarm_size: int = 30,
        iterations: int = 80,
        inertia: float = 0.7,
        cognitive_weight: float = 1.4,
        social_weight: float = 1.4,
        objective_weights: dict[str, float] | None = None,
        seed_positions: list[np.ndarray] | None = None,
    ):
        """Store the simulation scenario and PSO settings."""
        self.simulation = simulation
        self.users = users
        self.initial_drones = initial_drones
        self.priority_zones = priority_zones or []
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.inertia = inertia
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.objective_weights = objective_weights or OBJECTIVE_WEIGHTS
        self.seed_positions = seed_positions or []
        self.rng = np.random.default_rng(simulation.config.random_seed + 1)

    def optimize(self) -> tuple[np.ndarray, float, list[float]]:
        """Return the best UAV placement, final score, and score history."""
        positions = self._initialize_particles()
        search_steps = self.rng.normal(0.0, 25.0, positions.shape)

        personal_best_positions = positions.copy()
        personal_best_scores = np.array([self._objective(particle) for particle in positions])
        global_best_index = int(np.argmax(personal_best_scores))
        global_best_position = personal_best_positions[global_best_index].copy()
        global_best_score = float(personal_best_scores[global_best_index])
        score_history = [global_best_score]

        for _ in range(self.iterations):
            random_personal = self.rng.random(positions.shape)
            random_global = self.rng.random(positions.shape)
            # In PSO, this is search velocity, not real physical UAV movement.
            search_steps = (
                self.inertia * search_steps
                + self.cognitive_weight * random_personal * (personal_best_positions - positions)
                + self.social_weight * random_global * (global_best_position - positions)
            )

            positions = self.simulation.clip_positions(positions + search_steps)
            scores = np.array([self._objective(particle) for particle in positions])

            improved = scores > personal_best_scores
            personal_best_positions[improved] = positions[improved]
            personal_best_scores[improved] = scores[improved]

            best_index = int(np.argmax(personal_best_scores))
            if personal_best_scores[best_index] > global_best_score:
                global_best_score = float(personal_best_scores[best_index])
                global_best_position = personal_best_positions[best_index].copy()

            score_history.append(global_best_score)

        return global_best_position, global_best_score, score_history

    def _initialize_particles(self) -> np.ndarray:
        """Create candidate UAV placements for the swarm."""
        config = self.simulation.config
        particles = self.rng.uniform(
            low=[0.0, 0.0],
            high=[config.area_width, config.area_height],
            size=(self.swarm_size, config.num_drones, 2),
        )
        particles[0] = self.initial_drones
        for index, seed in enumerate(self.seed_positions, start=1):
            if index >= self.swarm_size:
                break
            particles[index] = self.simulation.clip_positions(seed)
        return particles

    def _objective(self, drones: np.ndarray) -> float:
        """Score one UAV placement for PSO."""
        config = self.simulation.config
        graph = self.simulation.build_network_graph(self.users, drones)

        connected_users = self.simulation.count_connected_users(self.users, drones)
        connected_ratio = connected_users / config.num_users

        drone_nodes = [f"drone_{index}" for index in range(config.num_drones)]
        relay_edges = graph.subgraph(drone_nodes).number_of_edges()
        max_relay_edges = max(1, config.num_drones * (config.num_drones - 1) / 2)
        relay_ratio = relay_edges / max_relay_edges

        coverage_ratio = self._coverage_ratio(drones, grid_size=25)
        priority_ratio = self.simulation.priority_coverage_ratio(drones, self.priority_zones)
        movement_cost = float(np.sum(np.linalg.norm(drones - self.initial_drones, axis=1)))
        normalized_movement = movement_cost / (
            config.num_drones * np.hypot(config.area_width, config.area_height)
        )

        # The positive terms reward useful coverage/connectivity. Movement is a
        # penalty because a solution that moves UAVs too far is less practical.
        return (
            self.objective_weights["connected_users"] * connected_ratio
            + self.objective_weights["priority_coverage"] * priority_ratio
            + self.objective_weights["area_coverage"] * coverage_ratio
            + self.objective_weights["relay_links"] * relay_ratio
            - self.objective_weights["movement_cost"] * normalized_movement
        )

    def _coverage_ratio(self, drones: np.ndarray, grid_size: int = 25) -> float:
        """Estimate area coverage using sampled grid points."""
        config = self.simulation.config
        ground_radius = self.simulation.ground_coverage_radius()
        x_values = np.linspace(0.0, config.area_width, grid_size)
        y_values = np.linspace(0.0, config.area_height, grid_size)
        points = np.array(np.meshgrid(x_values, y_values)).T.reshape(-1, 2)
        distances = np.linalg.norm(points[:, np.newaxis, :] - drones[np.newaxis, :, :], axis=2)
        obstacles = self.simulation.generate_obstacles()
        covered_points = 0
        for point, point_distances in zip(points, distances):
            covered = False
            for drone, distance in zip(drones, point_distances):
                if distance <= ground_radius and not self.simulation.is_link_blocked(
                    point, drone, obstacles
                ):
                    covered = True
                    break
            if covered:
                covered_points += 1
        return float(covered_points / len(points))


def kmeans_drone_placement(
    simulation: DisasterNetworkSimulation,
    users: np.ndarray,
    initial_drones: np.ndarray,
    iterations: int = 25,
) -> np.ndarray:
    """Place UAVs at user-cluster centres using a simple NumPy K-Means baseline."""
    config = simulation.config
    rng = np.random.default_rng(config.random_seed + 2)
    drone_count = config.num_drones

    if len(users) == 0:
        return initial_drones.copy()

    if len(users) >= drone_count:
        chosen_indices = rng.choice(len(users), size=drone_count, replace=False)
        centers = users[chosen_indices].copy()
    else:
        centers = initial_drones.copy()
        centers[: len(users)] = users

    for _ in range(iterations):
        distances = np.linalg.norm(users[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()

        for center_index in range(drone_count):
            cluster_users = users[labels == center_index]
            if len(cluster_users) > 0:
                new_centers[center_index] = cluster_users.mean(axis=0)

        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    return simulation.clip_positions(centers)
