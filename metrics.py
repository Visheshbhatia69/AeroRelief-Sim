"""Calculate metrics for a UAV placement."""

import networkx as nx
import numpy as np

from simulation import DisasterNetworkSimulation, PriorityZone, SimulationConfig


def evaluate_solution(
    config: SimulationConfig,
    users: np.ndarray,
    drones: np.ndarray,
    initial_drones: np.ndarray,
    graph: nx.Graph,
    priority_zones: list[PriorityZone] | None = None,
) -> dict[str, float]:
    """Return the main metrics used in the dissertation simulator."""
    connected_users = _connected_user_count(config, graph)
    coverage_percent = _coverage_percent(config, drones)
    priority_coverage_percent = _priority_coverage_percent(
        config, drones, priority_zones or []
    )
    average_path_length = _average_path_length(graph)
    movement_cost = float(np.sum(np.linalg.norm(drones - initial_drones, axis=1)))

    return {
        "connected_users_percent": 100.0 * connected_users / len(users),
        "coverage_percent": coverage_percent,
        "priority_coverage_percent": priority_coverage_percent,
        "connected_components": float(nx.number_connected_components(graph)),
        "average_path_length": average_path_length,
        "movement_cost": movement_cost,
    }


def _connected_user_count(config: SimulationConfig, graph: nx.Graph) -> int:
    """Count users that belong to a graph component containing at least one UAV."""
    drone_nodes = {f"drone_{index}" for index in range(config.num_drones)}
    connected_users = 0

    for component in nx.connected_components(graph):
        if component.intersection(drone_nodes):
            connected_users += sum(1 for node in component if node.startswith("user_"))

    return connected_users


def _coverage_percent(config: SimulationConfig, drones: np.ndarray, grid_size: int = 60) -> float:
    """Estimate map coverage by checking a regular grid of sample points."""
    simulation = DisasterNetworkSimulation(config)
    ground_radius = _ground_coverage_radius(config)
    x_values = np.linspace(0.0, config.area_width, grid_size)
    y_values = np.linspace(0.0, config.area_height, grid_size)
    points = np.array(np.meshgrid(x_values, y_values)).T.reshape(-1, 2)

    distances = np.linalg.norm(points[:, np.newaxis, :] - drones[np.newaxis, :, :], axis=2)
    obstacles = simulation.generate_obstacles()
    covered_points = 0
    for point, point_distances in zip(points, distances):
        covered = False
        for drone, distance in zip(drones, point_distances):
            if distance <= ground_radius and not simulation.is_link_blocked(point, drone, obstacles):
                covered = True
                break
        if covered:
            covered_points += 1
    covered_ratio = covered_points / len(points)
    return float(100.0 * covered_ratio)


def _priority_coverage_percent(
    config: SimulationConfig, drones: np.ndarray, priority_zones: list[PriorityZone]
) -> float:
    """Measure weighted coverage of important emergency zones."""
    if not priority_zones:
        return 0.0

    simulation = DisasterNetworkSimulation(config)
    covered_weight = 0.0
    total_weight = sum(zone.weight for zone in priority_zones)
    ground_radius = _ground_coverage_radius(config)
    obstacles = simulation.generate_obstacles()
    for zone in priority_zones:
        distances = np.linalg.norm(drones - np.array(zone.center), axis=1)
        covered = False
        for drone, distance in zip(drones, distances):
            if distance <= ground_radius + zone.radius and not simulation.is_link_blocked(
                np.array(zone.center), drone, obstacles
            ):
                covered = True
                break
        if covered:
            covered_weight += zone.weight

    return float(100.0 * covered_weight / total_weight)


def _ground_coverage_radius(config: SimulationConfig) -> float:
    """Return ground coverage radius from 3D range and UAV altitude."""
    if config.user_range <= config.uav_altitude:
        return 0.0
    return float(np.sqrt(config.user_range**2 - config.uav_altitude**2))


def _average_path_length(graph: nx.Graph) -> float:
    """Return average path length in the largest connected component."""
    if graph.number_of_nodes() == 0:
        return 0.0

    largest_component = max(nx.connected_components(graph), key=len)
    if len(largest_component) <= 1:
        return 0.0

    subgraph = graph.subgraph(largest_component)
    return float(nx.average_shortest_path_length(subgraph))
