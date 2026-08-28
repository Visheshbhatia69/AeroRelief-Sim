
from pathlib import Path

from experiment import ExperimentResult, run_experiment
from simulation import SimulationConfig
from visualization import (
    plot_3d_deployment,
    plot_comparison,
    plot_efficiency_chart,
    plot_metric_charts,
    plot_pso_convergence,
)


def main() -> None:
    """Run the command-line demo and save the main result figures."""
    config = SimulationConfig(
        area_width=1000.0,
        area_height=1000.0,
        num_users=80,
        num_drones=5,
        uav_altitude=120.0,
        user_range=180.0,
        drone_range=350.0,
        random_seed=42,
        scenario_type="Random users",
        priority_source="Synthetic demo zones",
    )

    result = run_experiment(config)

    print_summary(result)

    output_path = Path("results_comparison.png")
    plot_comparison(
        config=result.config,
        users=result.users,
        baseline_drones=result.baseline_drones,
        optimized_drones=result.optimized_drones,
        baseline_graph=result.baseline_graph,
        optimized_graph=result.optimized_graph,
        output_path=output_path,
        priority_zones=result.priority_zones,
        kmeans_drones=result.kmeans_drones,
        kmeans_graph=result.kmeans_graph,
    )
    print(f"\nSaved visualization to: {output_path.resolve()}")

    metrics_path = Path("results_metrics.png")
    plot_metric_charts(
        result.baseline_metrics,
        result.optimized_metrics,
        metrics_path,
        kmeans_metrics=result.kmeans_metrics,
    )
    print(f"Saved metric charts to: {metrics_path.resolve()}")

    efficiency_path = Path("results_efficiency.png")
    plot_efficiency_chart(result.optimized_metrics, efficiency_path)
    print(f"Saved efficiency chart to: {efficiency_path.resolve()}")

    convergence_path = Path("results_pso_convergence.png")
    plot_pso_convergence(result.pso_score_history, convergence_path)
    print(f"Saved PSO convergence chart to: {convergence_path.resolve()}")

    deployment_3d_path = Path("results_3d_deployment.png")
    plot_3d_deployment(
        config=result.config,
        users=result.users,
        baseline_drones=result.baseline_drones,
        kmeans_drones=result.kmeans_drones,
        optimized_drones=result.optimized_drones,
        output_path=deployment_3d_path,
        priority_zones=result.priority_zones,
    )
    print(f"Saved 3D deployment chart to: {deployment_3d_path.resolve()}")


def print_summary(result: ExperimentResult) -> None:
    """Print the setup and metrics in a presentation-friendly format."""
    config = result.config
    print("\nAeroRelief-Sim: Priority-Aware UAV Placement Simulator")
    print("Temporary Disaster Communication Support")
    print("=" * 45)
    print(f"Users: {config.num_users}")
    print(f"Drones: {config.num_drones}")
    print(f"Area: {config.area_width:.0f}m x {config.area_height:.0f}m")
    print(f"UAV altitude: {config.uav_altitude:.0f}m")
    print(f"Priority zones: {', '.join(zone.name for zone in result.priority_zones)}")
    print(f"Best PSO objective score: {result.best_score:.3f}")

    print_metrics("Baseline random/static placement", result.baseline_metrics)
    print_metrics("K-Means user-cluster placement", result.kmeans_metrics)
    print_metrics("Optimized PSO placement", result.optimized_metrics)
    print_improvements(result.baseline_metrics, result.optimized_metrics)


def print_metrics(title: str, values: dict[str, float]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Connected users:        {values['connected_users_percent']:.2f}%")
    print(f"Coverage percentage:    {values['coverage_percent']:.2f}%")
    print(f"Priority coverage:      {values['priority_coverage_percent']:.2f}%")
    print(f"Connected components:   {values['connected_components']:.0f}")
    print(f"Average path length:    {values['average_path_length']:.2f}")
    print(f"Movement cost:          {values['movement_cost']:.2f} m")
    if values["movement_cost"] > 0:
        print(f"User gain efficiency:   {values['connected_gain_per_100m']:.2f} pp/100m")
        print(f"Priority efficiency:    {values['priority_gain_per_100m']:.2f} pp/100m")


def print_improvements(baseline: dict[str, float], optimized: dict[str, float]) -> None:
    """Print the headline gains over the random/static baseline."""
    connected_gain = (
        optimized["connected_users_percent"] - baseline["connected_users_percent"]
    )
    priority_gain = (
        optimized["priority_coverage_percent"] - baseline["priority_coverage_percent"]
    )
    component_drop = baseline["connected_components"] - optimized["connected_components"]

    print("\nImprovement summary")
    print("-------------------")
    print(f"Connected users gain:   {connected_gain:.2f} percentage points")
    print(f"Priority coverage gain: {priority_gain:.2f} percentage points")
    print(f"Component reduction:    {component_drop:.0f} fewer components")


if __name__ == "__main__":
    main()
