"""Experiment pipeline for comparing UAV placement methods."""

from dataclasses import dataclass

import networkx as nx
import numpy as np

from metrics import evaluate_solution
from optimization import ParticleSwarmOptimizer, kmeans_drone_placement
from simulation import DisasterNetworkSimulation, PriorityZone, SimulationConfig


@dataclass
class ExperimentResult:
    """Container for the data, graphs and metrics from one run."""

    config: SimulationConfig
    users: np.ndarray
    baseline_drones: np.ndarray
    kmeans_drones: np.ndarray
    optimized_drones: np.ndarray
    priority_zones: list[PriorityZone]
    baseline_graph: nx.Graph
    kmeans_graph: nx.Graph
    optimized_graph: nx.Graph
    baseline_metrics: dict[str, float]
    kmeans_metrics: dict[str, float]
    optimized_metrics: dict[str, float]
    best_score: float
    pso_score_history: list[float]


def run_experiment(
    config: SimulationConfig,
    swarm_size: int = 45,
    iterations: int = 120,
    objective_weights: dict[str, float] | None = None,
) -> ExperimentResult:
    """Run random/static, K-Means and PSO on the same generated scenario."""
    simulation = DisasterNetworkSimulation(config)

    users = simulation.generate_users()
    baseline_drones = simulation.generate_drones()
    priority_zones = simulation.generate_priority_zones()
    kmeans_drones = kmeans_drone_placement(simulation, users, baseline_drones)

    optimizer = ParticleSwarmOptimizer(
        simulation=simulation,
        users=users,
        initial_drones=baseline_drones,
        priority_zones=priority_zones,
        swarm_size=swarm_size,
        iterations=iterations,
        objective_weights=objective_weights,
        seed_positions=[kmeans_drones],
    )
    optimized_drones, best_score, pso_score_history = optimizer.optimize()

    baseline_graph = simulation.build_network_graph(users, baseline_drones)
    kmeans_graph = simulation.build_network_graph(users, kmeans_drones)
    optimized_graph = simulation.build_network_graph(users, optimized_drones)

    baseline_metrics = evaluate_solution(
        config=config,
        users=users,
        drones=baseline_drones,
        initial_drones=baseline_drones,
        graph=baseline_graph,
        priority_zones=priority_zones,
    )
    kmeans_metrics = evaluate_solution(
        config=config,
        users=users,
        drones=kmeans_drones,
        initial_drones=baseline_drones,
        graph=kmeans_graph,
        priority_zones=priority_zones,
    )
    optimized_metrics = evaluate_solution(
        config=config,
        users=users,
        drones=optimized_drones,
        initial_drones=baseline_drones,
        graph=optimized_graph,
        priority_zones=priority_zones,
    )
    add_improvement_metrics(baseline_metrics, kmeans_metrics)
    add_improvement_metrics(baseline_metrics, optimized_metrics)

    return ExperimentResult(
        config=config,
        users=users,
        baseline_drones=baseline_drones,
        kmeans_drones=kmeans_drones,
        optimized_drones=optimized_drones,
        priority_zones=priority_zones,
        baseline_graph=baseline_graph,
        kmeans_graph=kmeans_graph,
        optimized_graph=optimized_graph,
        baseline_metrics=baseline_metrics,
        kmeans_metrics=kmeans_metrics,
        optimized_metrics=optimized_metrics,
        best_score=best_score,
        pso_score_history=pso_score_history,
    )


def add_improvement_metrics(
    baseline_metrics: dict[str, float], optimized_metrics: dict[str, float]
) -> None:
    """Add improvement values relative to the random/static baseline."""
    movement_cost = optimized_metrics["movement_cost"]
    connected_gain = (
        optimized_metrics["connected_users_percent"]
        - baseline_metrics["connected_users_percent"]
    )
    priority_gain = (
        optimized_metrics["priority_coverage_percent"]
        - baseline_metrics["priority_coverage_percent"]
    )

    optimized_metrics["connected_users_gain_pp"] = connected_gain
    optimized_metrics["priority_coverage_gain_pp"] = priority_gain

    if movement_cost > 0:
        optimized_metrics["connected_gain_per_100m"] = connected_gain / movement_cost * 100.0
        optimized_metrics["priority_gain_per_100m"] = priority_gain / movement_cost * 100.0
    else:
        optimized_metrics["connected_gain_per_100m"] = 0.0
        optimized_metrics["priority_gain_per_100m"] = 0.0

    baseline_metrics["connected_users_gain_pp"] = 0.0
    baseline_metrics["priority_coverage_gain_pp"] = 0.0
    baseline_metrics["connected_gain_per_100m"] = 0.0
    baseline_metrics["priority_gain_per_100m"] = 0.0
