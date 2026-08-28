
"""Matplotlib plots used by the command-line run and dashboard."""

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from simulation import DisasterNetworkSimulation, Obstacle, PriorityZone, SimulationConfig


def plot_comparison(
    config: SimulationConfig,
    users: np.ndarray,
    baseline_drones: np.ndarray,
    optimized_drones: np.ndarray,
    baseline_graph: nx.Graph,
    optimized_graph: nx.Graph,
    output_path: Path,
    priority_zones: list[PriorityZone] | None = None,
    kmeans_drones: np.ndarray | None = None,
    kmeans_graph: nx.Graph | None = None,
) -> None:
    """Save the spatial comparison of random/static, K-Means and PSO."""
    if kmeans_drones is None or kmeans_graph is None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
        plot_items = [
            (
                axes[0],
                baseline_drones,
                baseline_graph,
                "Baseline: random/static UAV placement",
                "#d95f02",
            ),
            (
                axes[1],
                optimized_drones,
                optimized_graph,
                "Optimized: PSO UAV placement",
                "#1b9e77",
            ),
        ]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        plot_items = [
            (
                axes[0],
                baseline_drones,
                baseline_graph,
                "Baseline: random/static",
                "#d95f02",
            ),
            (
                axes[1],
                kmeans_drones,
                kmeans_graph,
                "K-Means: user-cluster placement",
                "#4c78a8",
            ),
            (
                axes[2],
                optimized_drones,
                optimized_graph,
                "PSO: objective-optimized",
                "#1b9e77",
            ),
        ]

    for axis, drones, graph, title, color in plot_items:
        _plot_network(
            axis=axis,
            config=config,
            users=users,
            drones=drones,
            graph=graph,
            priority_zones=priority_zones or [],
            title=title,
            drone_color=color,
        )

    fig.suptitle("UAV Relay Placement for Temporary Disaster Communication", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metric_charts(
    baseline_metrics: dict[str, float],
    optimized_metrics: dict[str, float],
    output_path: Path,
    kmeans_metrics: dict[str, float] | None = None,
) -> None:
    """Save the main metric chart, using higher-is-better views where possible."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    all_metrics = [baseline_metrics, optimized_metrics]
    if kmeans_metrics is not None:
        all_metrics.insert(1, kmeans_metrics)
    cohesion_scores = _relative_cohesion_scores(all_metrics)
    service_scores = [_priority_service_score(metrics, cohesion) for metrics, cohesion in zip(all_metrics, cohesion_scores)]

    _bar_pair(
        axes[0, 0],
        "Connected Users (%)",
        baseline_metrics["connected_users_percent"],
        optimized_metrics["connected_users_percent"],
        ylabel="Users connected (%)",
        kmeans_value=None if kmeans_metrics is None else kmeans_metrics["connected_users_percent"],
    )
    _bar_pair(
        axes[0, 1],
        "Priority-Zone Coverage (%)",
        baseline_metrics["priority_coverage_percent"],
        optimized_metrics["priority_coverage_percent"],
        ylabel="Weighted priority coverage (%)",
        kmeans_value=None if kmeans_metrics is None else kmeans_metrics["priority_coverage_percent"],
    )
    _bar_pair(
        axes[1, 0],
        "Network Cohesion Score (%)",
        cohesion_scores[0],
        cohesion_scores[-1],
        ylabel="Relative cohesion score (%)",
        kmeans_value=None if kmeans_metrics is None else cohesion_scores[1],
    )
    _bar_pair(
        axes[1, 1],
        "Priority-Aware Service Score (%)",
        service_scores[0],
        service_scores[-1],
        ylabel="Combined service score (%)",
        kmeans_value=None if kmeans_metrics is None else service_scores[1],
    )

    fig.suptitle("Random vs K-Means vs PSO: Main Metrics (Higher is Better)", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _relative_cohesion_scores(metrics_list: list[dict[str, float]]) -> list[float]:
    """Turn fragmentation counts into a score that is easier to compare visually."""
    components = np.array([metrics["connected_components"] for metrics in metrics_list], dtype=float)
    worst = float(components.max())
    best = float(components.min())
    if worst == best:
        return [100.0 for _ in components]
    return [float(100.0 * (worst - value) / (worst - best)) for value in components]


def _priority_service_score(metrics: dict[str, float], cohesion_score: float) -> float:
    """Combine key service metrics into one dashboard-style summary score."""
    return float(
        0.40 * metrics["connected_users_percent"]
        + 0.35 * metrics["priority_coverage_percent"]
        + 0.15 * metrics["coverage_percent"]
        + 0.10 * cohesion_score
    )


def plot_efficiency_chart(
    optimized_metrics: dict[str, float],
    output_path: Path,
) -> None:
    """Save movement cost and gain-per-distance figures."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    axes[0].bar(["PSO movement"], [optimized_metrics["movement_cost"]], color="#1b9e77")
    axes[0].set_title("UAV Repositioning Cost")
    axes[0].set_ylabel("Total distance moved (m)")
    axes[0].grid(axis="y", linestyle=":", alpha=0.35)

    labels = ["Connected-user gain", "Priority-coverage gain"]
    values = [
        optimized_metrics["connected_gain_per_100m"],
        optimized_metrics["priority_gain_per_100m"],
    ]
    axes[1].bar(labels, values, color=["#4c78a8", "#9467bd"])
    axes[1].set_title("Movement Efficiency")
    axes[1].set_ylabel("Percentage-point gain per 100m moved")
    axes[1].tick_params(axis="x", rotation=12)
    axes[1].grid(axis="y", linestyle=":", alpha=0.35)

    fig.suptitle("UAV Movement Cost and Efficiency", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_pso_convergence(score_history: list[float], output_path: Path) -> None:
    """Save the best PSO score seen after each iteration."""
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.plot(range(len(score_history)), score_history, color="#1b9e77", linewidth=2)
    axis.set_title("PSO Convergence")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Best objective score")
    axis.grid(True, linestyle=":", alpha=0.4)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_3d_deployment(
    config: SimulationConfig,
    users: np.ndarray,
    baseline_drones: np.ndarray,
    kmeans_drones: np.ndarray,
    optimized_drones: np.ndarray,
    output_path: Path,
    priority_zones: list[PriorityZone] | None = None,
) -> None:
    """Save a simple 3D view with ground users and fixed-altitude UAVs."""
    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    plot_items = [
        ("Random/static", baseline_drones, "#d95f02"),
        ("K-Means", kmeans_drones, "#4c78a8"),
        ("PSO optimized", optimized_drones, "#1b9e77"),
    ]

    for index, (title, drones, color) in enumerate(plot_items, start=1):
        axis = fig.add_subplot(1, 3, index, projection="3d")
        _plot_3d_network(
            axis=axis,
            config=config,
            users=users,
            drones=drones,
            priority_zones=priority_zones or [],
            title=title,
            drone_color=color,
        )

    fig.suptitle("3D UAV Deployment View: ground users and fixed-altitude UAVs")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _bar_pair(
    axis: plt.Axes,
    title: str,
    baseline_value: float,
    optimized_value: float,
    ylabel: str,
    kmeans_value: float | None = None,
) -> None:
    """Draw one small method-comparison bar chart."""
    labels = ["Random/static", "PSO optimized"]
    values = [baseline_value, optimized_value]
    colors = ["#d95f02", "#1b9e77"]
    if kmeans_value is not None:
        labels = ["Random/static", "K-Means", "PSO optimized"]
        values = [baseline_value, kmeans_value, optimized_value]
        colors = ["#d95f02", "#4c78a8", "#1b9e77"]

    axis.bar(
        labels,
        values,
        color=colors,
    )
    for index, value in enumerate(values):
        axis.text(index, value + max(values) * 0.025, f"{value:.1f}", ha="center", fontsize=9)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_ylim(0, max(values) * 1.16 if max(values) > 0 else 1)
    axis.grid(axis="y", linestyle=":", alpha=0.35)


def _plot_network(
    axis: plt.Axes,
    config: SimulationConfig,
    users: np.ndarray,
    drones: np.ndarray,
    graph: nx.Graph,
    priority_zones: list[PriorityZone],
    title: str,
    drone_color: str,
) -> None:
    """Draw users, UAVs, zones, obstacles and communication links on one map."""
    axis.set_title(title)
    axis.set_xlim(0.0, config.area_width)
    axis.set_ylim(0.0, config.area_height)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x position (m)")
    axis.set_ylabel("y position (m)")
    axis.grid(True, linestyle=":", alpha=0.35)

    _draw_priority_zones(axis, priority_zones)
    _draw_obstacles_2d(axis, DisasterNetworkSimulation(config).generate_obstacles())

    ground_radius = (
        float(np.sqrt(config.user_range**2 - config.uav_altitude**2))
        if config.user_range > config.uav_altitude
        else 0.0
    )

    for drone in drones:
        coverage_circle = plt.Circle(
            drone,
            ground_radius,
            color=drone_color,
            alpha=0.08,
            linewidth=0,
        )
        axis.add_patch(coverage_circle)

    _draw_edges(axis, graph)

    axis.scatter(
        users[:, 0],
        users[:, 1],
        s=22,
        c="#4c78a8",
        alpha=0.85,
        label="Users/rescue nodes",
    )
    axis.scatter(
        drones[:, 0],
        drones[:, 1],
        s=145,
        c=drone_color,
        marker="^",
        edgecolors="black",
        linewidths=0.8,
        label="UAV relay nodes",
    )

    for index, drone in enumerate(drones):
        axis.text(drone[0] + 10, drone[1] + 10, f"D{index}", fontsize=8, weight="bold")

    axis.legend(loc="upper right", fontsize=8)


def _plot_3d_network(
    axis: plt.Axes,
    config: SimulationConfig,
    users: np.ndarray,
    drones: np.ndarray,
    priority_zones: list[PriorityZone],
    title: str,
    drone_color: str,
) -> None:
    """Draw one 3D deployment subplot for a placement method."""
    altitude = config.uav_altitude
    ground_radius = (
        float(np.sqrt(config.user_range**2 - altitude**2))
        if config.user_range > altitude
        else 0.0
    )

    axis.set_title(title)
    axis.set_xlim(0.0, config.area_width)
    axis.set_ylim(0.0, config.area_height)
    axis.set_zlim(0.0, max(altitude * 1.35, 1.0))
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("altitude (m)")

    _draw_3d_ground(axis, config)
    _draw_3d_obstacles(axis, DisasterNetworkSimulation(config).generate_obstacles())
    _draw_3d_priority_zones(axis, priority_zones)

    axis.scatter(
        users[:, 0],
        users[:, 1],
        np.zeros(len(users)),
        s=14,
        c="#4c78a8",
        alpha=0.75,
        label="Ground users",
    )
    axis.scatter(
        drones[:, 0],
        drones[:, 1],
        np.full(len(drones), altitude),
        s=80,
        c=drone_color,
        marker="^",
        edgecolors="black",
        linewidths=0.6,
        label="UAVs",
    )

    for drone in drones:
        axis.plot(
            [drone[0], drone[0]],
            [drone[1], drone[1]],
            [0.0, altitude],
            color=drone_color,
            alpha=0.35,
            linewidth=1.0,
        )
        _draw_3d_coverage_circle(axis, drone, ground_radius, drone_color)

    axis.view_init(elev=24, azim=-58)
    axis.legend(loc="upper left", fontsize=7)


def _draw_3d_ground(axis: plt.Axes, config: SimulationConfig) -> None:
    """Draw the disaster-area ground plane."""
    vertices = [
        [
            (0.0, 0.0, 0.0),
            (config.area_width, 0.0, 0.0),
            (config.area_width, config.area_height, 0.0),
            (0.0, config.area_height, 0.0),
        ]
    ]
    plane = Poly3DCollection(vertices, alpha=0.08, facecolor="#999999", edgecolor="#666666")
    axis.add_collection3d(plane)


def _draw_3d_priority_zones(axis: plt.Axes, priority_zones: list[PriorityZone]) -> None:
    """Draw priority zones on the ground plane."""
    for zone in priority_zones:
        color = _priority_zone_color(zone.name)
        angles = np.linspace(0.0, 2.0 * np.pi, 80)
        x_values = zone.center[0] + zone.radius * np.cos(angles)
        y_values = zone.center[1] + zone.radius * np.sin(angles)
        z_values = np.zeros_like(x_values)
        axis.plot(x_values, y_values, z_values, color=color, linewidth=1.2, alpha=0.9)
        axis.scatter(
            [zone.center[0]],
            [zone.center[1]],
            [0.0],
            marker="*",
            s=70,
            c=color,
            edgecolors="black",
            linewidths=0.4,
        )


def _draw_3d_obstacles(axis: plt.Axes, obstacles: list[Obstacle]) -> None:
    """Draw obstacle blocks as simple cuboids."""
    for obstacle in obstacles:
        x0, x1 = obstacle.x_min, obstacle.x_max
        y0, y1 = obstacle.y_min, obstacle.y_max
        z0, z1 = 0.0, obstacle.height
        vertices = [
            [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
            [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
            [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
            [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
            [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)],
            [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)],
        ]
        block = Poly3DCollection(
            vertices,
            facecolor="#666666",
            edgecolor="#333333",
            alpha=0.24,
            linewidth=0.5,
        )
        axis.add_collection3d(block)


def _draw_3d_coverage_circle(
    axis: plt.Axes, drone: np.ndarray, ground_radius: float, color: str
) -> None:
    """Draw one UAV's ground coverage footprint."""
    if ground_radius <= 0.0:
        return
    angles = np.linspace(0.0, 2.0 * np.pi, 80)
    x_values = drone[0] + ground_radius * np.cos(angles)
    y_values = drone[1] + ground_radius * np.sin(angles)
    z_values = np.zeros_like(x_values)
    axis.plot(x_values, y_values, z_values, color=color, linewidth=0.9, alpha=0.45)


def _draw_priority_zones(axis: plt.Axes, priority_zones: list[PriorityZone]) -> None:
    """Draw labelled priority zones on the 2D map."""
    for zone in priority_zones:
        color = _priority_zone_color(zone.name)
        label = _short_priority_label(zone.name)
        circle = plt.Circle(
            zone.center,
            zone.radius,
            color=color,
            alpha=0.16,
            linewidth=1.5,
            fill=True,
        )
        outline = plt.Circle(
            zone.center,
            zone.radius,
            color=color,
            alpha=0.9,
            linewidth=1.2,
            fill=False,
        )
        axis.add_patch(circle)
        axis.add_patch(outline)
        axis.scatter(
            [zone.center[0]],
            [zone.center[1]],
            marker="*",
            s=120,
            c=color,
            edgecolors="black",
            linewidths=0.6,
            zorder=4,
        )
        axis.text(
            zone.center[0] + 12,
            zone.center[1] + 12,
            f"{label} (w={zone.weight:g})",
            fontsize=8,
            weight="bold",
            color=color,
        )


def _draw_obstacles_2d(axis: plt.Axes, obstacles: list[Obstacle]) -> None:
    """Draw the ground footprint of each obstacle."""
    for obstacle in obstacles:
        rectangle = plt.Rectangle(
            (obstacle.x_min, obstacle.y_min),
            obstacle.x_max - obstacle.x_min,
            obstacle.y_max - obstacle.y_min,
            facecolor="#666666",
            edgecolor="#333333",
            alpha=0.18,
            linewidth=1.0,
            zorder=0,
        )
        axis.add_patch(rectangle)
        axis.text(
            obstacle.x_min + 6,
            obstacle.y_min + 14,
            f"{obstacle.name}\n{obstacle.height:.0f}m",
            fontsize=6,
            color="#333333",
            zorder=5,
        )


def _priority_zone_color(zone_name: str) -> str:
    """Pick a stable colour from the zone name."""
    lower_name = zone_name.lower()
    if "hospital" in lower_name or "health" in lower_name or "doctor" in lower_name:
        return "#d62728"
    if "shelter" in lower_name or "school" in lower_name or "community" in lower_name:
        return "#9467bd"
    if "rescue" in lower_name or "fire" in lower_name:
        return "#ff7f0e"
    if "risk" in lower_name:
        return "#8c564b"
    if "police" in lower_name:
        return "#1f77b4"
    return "#444444"


def _short_priority_label(zone_name: str) -> str:
    """Trim long OSM names so map labels stay readable."""
    base_name = zone_name.split(" (", 1)[0]
    if len(base_name) <= 22:
        return base_name
    return f"{base_name[:19]}..."


def _draw_edges(axis: plt.Axes, graph: nx.Graph) -> None:
    """Draw graph edges as communication links."""
    for first_node, second_node in graph.edges:
        first_position = graph.nodes[first_node]["pos"]
        second_position = graph.nodes[second_node]["pos"]
        is_drone_link = first_node.startswith("drone_") and second_node.startswith("drone_")
        color = "#555555" if is_drone_link else "#9ecae1"
        width = 1.4 if is_drone_link else 0.7
        alpha = 0.65 if is_drone_link else 0.45
        axis.plot(
            [first_position[0], second_position[0]],
            [first_position[1], second_position[1]],
            color=color,
            linewidth=width,
            alpha=alpha,
            zorder=1,
        )
