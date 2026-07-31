
"""Create the disaster map, nodes, priority zones, obstacles, and graph links."""

import csv
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    """Basic settings for one simulation run."""

    area_width: float = 1000.0
    area_height: float = 1000.0
    num_users: int = 80
    num_drones: int = 5
    uav_altitude: float = 120.0
    user_range: float = 180.0
    drone_range: float = 350.0
    random_seed: int = 42
    scenario_type: str = "Random users"
    priority_source: str = "Synthetic demo zones"
    obstacles_enabled: bool = False


@dataclass(frozen=True)
class PriorityZone:
    """Important emergency area that should receive stronger communication support."""

    name: str
    center: tuple[float, float]
    radius: float
    weight: float


@dataclass(frozen=True)
class Obstacle:
    """Simple rectangular building/blocked area with height."""

    name: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    height: float


class DisasterNetworkSimulation:
    """Disaster communication simulator with fixed-altitude UAV links."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

    def generate_users(self) -> np.ndarray:
        """Create random user/rescue-worker positions."""
        if self.config.scenario_type == "Clustered users":
            return self._clustered_user_positions()
        if self.config.scenario_type == "Hotspot priority zones":
            return self._hotspot_user_positions()
        return self._random_positions(self.config.num_users)

    def generate_drones(self) -> np.ndarray:
        """Create random UAV positions for the baseline method."""
        return self._random_positions(self.config.num_drones)

    def generate_priority_zones(self) -> list[PriorityZone]:
        """Create emergency areas that are more important than normal map space."""
        if self.config.priority_source == "Real OSM Stirling zones":
            zones = self._load_osm_priority_zones()
            if zones:
                return zones

        width = self.config.area_width
        height = self.config.area_height
        return [
            PriorityZone("Hospital", (0.25 * width, 0.72 * height), 90.0, 4.0),
            PriorityZone("Shelter", (0.72 * width, 0.68 * height), 110.0, 3.0),
            PriorityZone("Rescue centre", (0.54 * width, 0.30 * height), 100.0, 3.5),
            PriorityZone("High-risk zone", (0.20 * width, 0.25 * height), 85.0, 2.5),
        ]

    def generate_obstacles(self) -> list[Obstacle]:
        """Create simplified physical obstacles for line-of-sight testing."""
        if not self.config.obstacles_enabled:
            return []

        width = self.config.area_width
        height = self.config.area_height
        return [
            Obstacle("Damaged building", 0.34 * width, 0.55 * height, 0.48 * width, 0.72 * height, 85.0),
            Obstacle("Collapsed high-rise", 0.58 * width, 0.38 * height, 0.72 * width, 0.56 * height, 110.0),
            Obstacle("Industrial obstruction", 0.18 * width, 0.36 * height, 0.32 * width, 0.52 * height, 70.0),
            Obstacle("Dense urban block", 0.62 * width, 0.70 * height, 0.82 * width, 0.84 * height, 60.0),
        ]

    def _load_osm_priority_zones(self) -> list[PriorityZone]:
        """Load real priority locations from a saved OpenStreetMap CSV."""
        path = Path("data") / "stirling_priority_zones_osm.csv"
        if not path.exists():
            return []

        with path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            return []

        latitudes = np.array([float(row["latitude"]) for row in rows])
        longitudes = np.array([float(row["longitude"]) for row in rows])
        min_latitude, max_latitude = float(latitudes.min()), float(latitudes.max())
        min_longitude, max_longitude = float(longitudes.min()), float(longitudes.max())
        latitude_span = max(max_latitude - min_latitude, 1e-9)
        longitude_span = max(max_longitude - min_longitude, 1e-9)

        zones = []
        for row in rows:
            latitude = float(row["latitude"])
            longitude = float(row["longitude"])
            # The simulator works in metres on a square map, so real latitude and
            # longitude values are scaled into the current experiment area.
            x_position = (
                (longitude - min_longitude) / longitude_span * self.config.area_width
            )
            y_position = (
                (latitude - min_latitude) / latitude_span * self.config.area_height
            )
            zones.append(
                PriorityZone(
                    name=f"{row['name']} ({row['amenity']})",
                    center=(x_position, y_position),
                    radius=float(row["radius_m"]),
                    weight=float(row["weight"]),
                )
            )

        return zones

    def priority_coverage_ratio(
        self, drones: np.ndarray, priority_zones: list[PriorityZone]
    ) -> float:
        """Measure weighted priority-zone coverage from 0 to 1."""
        if not priority_zones:
            return 0.0

        covered_weight = 0.0
        total_weight = sum(zone.weight for zone in priority_zones)
        ground_radius = self.ground_coverage_radius()
        obstacles = self.generate_obstacles()
        for zone in priority_zones:
            zone_center = np.array(zone.center)
            distances = np.linalg.norm(drones - zone_center, axis=1)
            covered = False
            for drone, distance in zip(drones, distances):
                # A zone counts as covered when the UAV footprint reaches the zone
                # circle and there is no simplified obstacle blocking the link.
                if distance <= ground_radius + zone.radius and not self.is_link_blocked(zone_center, drone, obstacles):
                    covered = True
                    break
            if covered:
                covered_weight += zone.weight

        return covered_weight / total_weight

    def build_network_graph(self, users: np.ndarray, drones: np.ndarray) -> nx.Graph:
        """Create a graph where nodes are users/UAVs and edges are communication links."""
        graph = nx.Graph()

        for user_index, position in enumerate(users):
            graph.add_node(f"user_{user_index}", node_type="user", pos=tuple(position))

        for drone_index, position in enumerate(drones):
            graph.add_node(f"drone_{drone_index}", node_type="drone", pos=tuple(position))

        self._add_user_drone_edges(graph, users, drones)
        self._add_drone_drone_edges(graph, drones)
        return graph

    def count_connected_users(self, users: np.ndarray, drones: np.ndarray) -> int:
        """Count users that are directly connected to at least one UAV."""
        distances = self.user_drone_distances(users, drones)
        obstacles = self.generate_obstacles()
        connected_users = 0
        for user_index, drone_index in np.argwhere(distances <= self.config.user_range):
            if not self.is_link_blocked(users[user_index], drones[drone_index], obstacles):
                connected_users += 1
                distances[user_index, :] = np.inf
        return connected_users

    def user_drone_distances(self, users: np.ndarray, drones: np.ndarray) -> np.ndarray:
        """Calculate all user-to-UAV 3D distances using fixed UAV altitude."""
        horizontal_distances = np.linalg.norm(
            users[:, np.newaxis, :] - drones[np.newaxis, :, :],
            axis=2,
        )
        return np.hypot(horizontal_distances, self.config.uav_altitude)

    def ground_coverage_radius(self) -> float:
        """Return the ground footprint radius for a UAV at the configured altitude."""
        if self.config.user_range <= self.config.uav_altitude:
            return 0.0
        return float(np.sqrt(self.config.user_range**2 - self.config.uav_altitude**2))

    def is_link_blocked(
        self,
        ground_point: np.ndarray,
        drone_point: np.ndarray,
        obstacles: list[Obstacle] | None = None,
    ) -> bool:
        """Return True if an obstacle intersects the ground-to-UAV line of sight."""
        active_obstacles = self.generate_obstacles() if obstacles is None else obstacles
        if not active_obstacles:
            return False

        ground = np.asarray(ground_point, dtype=float)
        drone = np.asarray(drone_point, dtype=float)
        for obstacle in active_obstacles:
            if self._line_blocked_by_obstacle(ground, drone, obstacle):
                return True
        return False

    def _line_blocked_by_obstacle(
        self, ground: np.ndarray, drone: np.ndarray, obstacle: Obstacle
    ) -> bool:
        """Approximate line-of-sight blockage using sampled points along the link."""
        # Sampling keeps this lightweight. A full radio simulator would use a more
        # detailed propagation and geometry model.
        for fraction in np.linspace(0.05, 0.95, 19):
            x_position = ground[0] + fraction * (drone[0] - ground[0])
            y_position = ground[1] + fraction * (drone[1] - ground[1])
            z_position = fraction * self.config.uav_altitude
            inside_rectangle = (
                obstacle.x_min <= x_position <= obstacle.x_max
                and obstacle.y_min <= y_position <= obstacle.y_max
            )
            if inside_rectangle and z_position <= obstacle.height:
                return True
        return False

    def clip_positions(self, positions: np.ndarray) -> np.ndarray:
        """Keep candidate UAV positions inside the map."""
        clipped = positions.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], 0.0, self.config.area_width)
        clipped[:, 1] = np.clip(clipped[:, 1], 0.0, self.config.area_height)
        return clipped

    def _random_positions(self, count: int) -> np.ndarray:
        """Return random x/y points inside the disaster area."""
        x_values = self.rng.uniform(0.0, self.config.area_width, count)
        y_values = self.rng.uniform(0.0, self.config.area_height, count)
        return np.column_stack((x_values, y_values))

    def _clustered_user_positions(self) -> np.ndarray:
        """Create users around a few rescue activity clusters."""
        centers = np.array(
            [
                [0.30 * self.config.area_width, 0.70 * self.config.area_height],
                [0.70 * self.config.area_width, 0.65 * self.config.area_height],
                [0.50 * self.config.area_width, 0.30 * self.config.area_height],
            ]
        )
        cluster_ids = self.rng.integers(0, len(centers), self.config.num_users)
        noise = self.rng.normal(0.0, 85.0, size=(self.config.num_users, 2))
        return self.clip_positions(centers[cluster_ids] + noise)

    def _hotspot_user_positions(self) -> np.ndarray:
        """Create more users near priority zones and some users elsewhere."""
        zones = self.generate_priority_zones()
        hotspot_count = int(self.config.num_users * 0.75)
        random_count = self.config.num_users - hotspot_count

        zone_centers = np.array([zone.center for zone in zones])
        zone_ids = self.rng.integers(0, len(zone_centers), hotspot_count)
        hotspot_noise = self.rng.normal(0.0, 70.0, size=(hotspot_count, 2))
        hotspot_users = zone_centers[zone_ids] + hotspot_noise
        random_users = self._random_positions(random_count)
        return self.clip_positions(np.vstack((hotspot_users, random_users)))

    def _add_user_drone_edges(
        self, graph: nx.Graph, users: np.ndarray, drones: np.ndarray
    ) -> None:
        """Connect each user to each UAV inside user communication range."""
        distances = self.user_drone_distances(users, drones)
        obstacles = self.generate_obstacles()
        for user_index, drone_index in np.argwhere(distances <= self.config.user_range):
            if self.is_link_blocked(users[user_index], drones[drone_index], obstacles):
                continue
            graph.add_edge(
                f"user_{user_index}",
                f"drone_{drone_index}",
                weight=float(distances[user_index, drone_index]),
            )

    def _add_drone_drone_edges(self, graph: nx.Graph, drones: np.ndarray) -> None:
        """Connect UAVs to other UAVs inside relay range."""
        for first_index in range(len(drones)):
            for second_index in range(first_index + 1, len(drones)):
                distance = float(np.linalg.norm(drones[first_index] - drones[second_index]))
                if distance <= self.config.drone_range:
                    graph.add_edge(
                        f"drone_{first_index}",
                        f"drone_{second_index}",
                        weight=distance,
                    )
