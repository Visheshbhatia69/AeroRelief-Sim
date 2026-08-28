"""Run repeated experiments for evaluation evidence.

The Streamlit dashboard is useful for inspection, but formal evaluation needs
results averaged across multiple random seeds. This script keeps those runs in
one place and writes the CSV/figure outputs used in analysis.
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment import run_experiment
from metrics import evaluate_solution
from optimization import OBJECTIVE_WEIGHTS, WEIGHT_PROFILES
from simulation import SimulationConfig
from visualization import plot_comparison


OUTPUT_DIR = Path("evaluation_outputs")

METHOD_LABELS = {
    "baseline": "Random/static",
    "kmeans": "K-Means",
    "pso": "PSO",
}

METHOD_COLORS = {
    "Random/static": "#d95f02",
    "K-Means": "#4c78a8",
    "PSO": "#1b9e77",
}


def relay_link_ratio(graph, num_drones: int) -> float:
    """Measure how many possible UAV-to-UAV relay links are active."""
    max_links = num_drones * (num_drones - 1) / 2
    if max_links <= 0:
        return 0.0

    links = 0
    for u, v in graph.edges():
        if str(u).startswith("drone_") and str(v).startswith("drone_"):
            links += 1
    return links / max_links


def objective_score(metrics: dict, graph, config: SimulationConfig) -> float:
    """Apply the same objective formula to any placement method."""
    diagonal = math.hypot(config.area_width, config.area_height)
    max_movement = max(config.num_drones * diagonal, 1.0)

    cu = metrics["connected_users_percent"] / 100.0
    pz = metrics["priority_coverage_percent"] / 100.0
    ac = metrics["coverage_percent"] / 100.0
    rl = relay_link_ratio(graph, config.num_drones)
    mc = metrics["movement_cost"] / max_movement

    return (
        OBJECTIVE_WEIGHTS["connected_users"] * cu
        + OBJECTIVE_WEIGHTS["priority_coverage"] * pz
        + OBJECTIVE_WEIGHTS["area_coverage"] * ac
        + OBJECTIVE_WEIGHTS["relay_links"] * rl
        - OBJECTIVE_WEIGHTS["movement_cost"] * mc
    )


def run_one(seed: int, config_kwargs: dict, swarm_size: int, iterations: int, weights=None) -> list[dict]:
    """Run one random/static, K-Means and PSO comparison for a seed."""
    config = SimulationConfig(random_seed=seed, **config_kwargs)
    result = run_experiment(
        config=config,
        swarm_size=swarm_size,
        iterations=iterations,
        objective_weights=weights,
    )

    rows = []
    method_data = [
        ("baseline", result.baseline_metrics, result.baseline_graph),
        ("kmeans", result.kmeans_metrics, result.kmeans_graph),
        ("pso", result.optimized_metrics, result.optimized_graph),
    ]

    for key, metrics, graph in method_data:
        row = {
            "seed": seed,
            "method": METHOD_LABELS[key],
            "connected_users_percent": metrics["connected_users_percent"],
            "priority_coverage_percent": metrics["priority_coverage_percent"],
            "area_coverage_percent": metrics["coverage_percent"],
            "connected_components": metrics["connected_components"],
            "average_path_length_hops": metrics["average_path_length"],
            "movement_cost_m": metrics["movement_cost"],
            "relay_link_ratio": relay_link_ratio(graph, config.num_drones),
            "objective_score": objective_score(metrics, graph, config),
        }
        rows.append(row)

    return rows


def run_many(name: str, seeds: list[int], config_kwargs: dict, swarm_size: int, iterations: int, weights=None) -> pd.DataFrame:
    """Run several seeds in parallel and save the raw CSV table."""
    all_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(run_one, seed, config_kwargs, swarm_size, iterations, weights)
            for seed in seeds
        ]
        for future in as_completed(futures):
            all_rows.extend(future.result())

    df = pd.DataFrame(all_rows).sort_values(["seed", "method"])
    df.to_csv(OUTPUT_DIR / f"{name}_raw_runs.csv", index=False)
    return df


def summary_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Calculate mean, standard deviation and 95% confidence intervals."""
    metric_cols = [
        "connected_users_percent",
        "priority_coverage_percent",
        "area_coverage_percent",
        "connected_components",
        "average_path_length_hops",
        "movement_cost_m",
        "relay_link_ratio",
        "objective_score",
    ]
    grouped = df.groupby("method")[metric_cols]
    summary = grouped.agg(["mean", "std", "count"])

    flat_rows = []
    for method in summary.index:
        row = {"method": method}
        for metric in metric_cols:
            mean = summary.loc[method, (metric, "mean")]
            std = summary.loc[method, (metric, "std")]
            count = summary.loc[method, (metric, "count")]
            ci95 = 1.96 * std / math.sqrt(count) if count > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = ci95
        flat_rows.append(row)

    out = pd.DataFrame(flat_rows)
    out.to_csv(OUTPUT_DIR / f"{name}_summary.csv", index=False)
    return out


def bar_with_ci(summary: pd.DataFrame, metric: str, title: str, ylabel: str, path: Path, higher_is_better=True) -> None:
    """Plot method means with 95% confidence intervals."""
    order = ["Random/static", "K-Means", "PSO"]
    data = summary.set_index("method").loc[order]
    means = data[f"{metric}_mean"]
    ci = data[f"{metric}_ci95"]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(order, means, yerr=ci, capsize=6, color=[METHOD_COLORS[m] for m in order], edgecolor="#222")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)

    if higher_is_better:
        ax.text(0.01, 0.96, "Higher is better", transform=ax.transAxes, fontsize=10, va="top")
    else:
        ax.text(0.01, 0.96, "Lower is better", transform=ax.transAxes, fontsize=10, va="top")

    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def default_result_figures(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Create the main repeated-run result figures."""
    chart_specs = [
        ("connected_users_percent", "Connected users across 30 seeds", "Connected users (%)", True),
        ("priority_coverage_percent", "Priority-zone coverage across 30 seeds", "Weighted priority coverage (%)", True),
        ("objective_score", "Common objective score across 30 seeds", "Weighted objective score", True),
        ("connected_components", "Network fragmentation across 30 seeds", "Connected components (count)", False),
    ]
    for metric, title, ylabel, high in chart_specs:
        bar_with_ci(summary, metric, title, ylabel, OUTPUT_DIR / f"default_{metric}.png", high)

    metrics = [
        "connected_users_percent",
        "priority_coverage_percent",
        "area_coverage_percent",
        "objective_score",
    ]
    labels = ["Connected users (%)", "Priority coverage (%)", "Area coverage (%)", "Objective score"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, metric, label in zip(axes.ravel(), metrics, labels):
        values = [df[df["method"] == method][metric] for method in ["Random/static", "K-Means", "PSO"]]
        bp = ax.boxplot(values, tick_labels=["Random", "K-Means", "PSO"], patch_artist=True)
        for patch, method in zip(bp["boxes"], ["Random/static", "K-Means", "PSO"]):
            patch.set_facecolor(METHOD_COLORS[method])
            patch.set_alpha(0.75)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Result spread across 30 independent random seeds", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "default_distribution_boxplots.png", dpi=220)
    plt.close(fig)


def run_uav_sensitivity() -> pd.DataFrame:
    """Check how the PSO result changes as more UAVs are available."""
    rows = []
    seeds = list(range(200, 210))
    for num_drones in range(3, 9):
        df = run_many(
            name=f"uav_{num_drones}",
            seeds=seeds,
            config_kwargs={"num_drones": num_drones},
            swarm_size=30,
            iterations=70,
        )
        pso = df[df["method"] == "PSO"].copy()
        pso["num_drones"] = num_drones
        rows.append(pso)

    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "uav_count_sensitivity_raw.csv", index=False)

    summary = combined.groupby("num_drones").agg(
        connected_users_mean=("connected_users_percent", "mean"),
        connected_users_std=("connected_users_percent", "std"),
        priority_coverage_mean=("priority_coverage_percent", "mean"),
        priority_coverage_std=("priority_coverage_percent", "std"),
        movement_cost_mean=("movement_cost_m", "mean"),
        objective_score_mean=("objective_score", "mean"),
    ).reset_index()
    summary.to_csv(OUTPUT_DIR / "uav_count_sensitivity_summary.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(9, 5.3))
    ax1.plot(summary["num_drones"], summary["connected_users_mean"], marker="o", label="Connected users (%)", color="#1f77b4")
    ax1.plot(summary["num_drones"], summary["priority_coverage_mean"], marker="s", label="Priority coverage (%)", color="#9467bd")
    ax1.set_xlabel("Number of UAV relay nodes")
    ax1.set_ylabel("Coverage performance (%)")
    ax1.set_ylim(0, 105)
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(summary["num_drones"], summary["movement_cost_mean"], marker="^", label="Movement cost (m)", color="#444")
    ax2.set_ylabel("Mean movement cost (m)")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="lower right")
    ax1.set_title("UAV-count sensitivity: performance gain versus repositioning cost")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "uav_count_sensitivity.png", dpi=220)
    plt.close(fig)

    return summary


def run_weight_sensitivity() -> pd.DataFrame:
    """Check whether conclusions change under different objective weights."""
    rows = []
    seeds = list(range(300, 312))
    for profile_name, weights in WEIGHT_PROFILES.items():
        df = run_many(
            name=f"weights_{profile_name.lower().replace(' ', '_')}",
            seeds=seeds,
            config_kwargs={},
            swarm_size=30,
            iterations=70,
            weights=weights,
        )
        pso = df[df["method"] == "PSO"].copy()
        pso["weight_profile"] = profile_name
        for key, value in weights.items():
            pso[f"{key}_weight"] = value
        rows.append(pso)

    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "weight_sensitivity_raw.csv", index=False)
    summary = combined.groupby("weight_profile").agg(
        connected_users_mean=("connected_users_percent", "mean"),
        priority_coverage_mean=("priority_coverage_percent", "mean"),
        movement_cost_mean=("movement_cost_m", "mean"),
        objective_score_mean=("objective_score", "mean"),
    ).reset_index()
    summary.to_csv(OUTPUT_DIR / "weight_sensitivity_summary.csv", index=False)

    profile_order = list(WEIGHT_PROFILES.keys())
    summary = summary.set_index("weight_profile").loc[profile_order].reset_index()
    x = np.arange(len(summary))
    width = 0.28

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width, summary["connected_users_mean"], width, label="Connected users (%)", color="#1f77b4")
    ax.bar(x, summary["priority_coverage_mean"], width, label="Priority coverage (%)", color="#9467bd")
    ax.bar(x + width, summary["objective_score_mean"] * 100, width, label="Objective score x100", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["weight_profile"], rotation=15, ha="right")
    ax.set_ylabel("Mean score / percentage")
    ax.set_title("Sensitivity analysis: objective weights change the trade-off")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "weight_sensitivity.png", dpi=220)
    plt.close(fig)

    return summary


def run_osm_example() -> None:
    """Run a small example using the OpenStreetMap priority-zone data."""
    seeds = list(range(500, 510))
    df = run_many(
        name="osm_priority_zones",
        seeds=seeds,
        config_kwargs={"priority_source": "Real OSM Stirling zones"},
        swarm_size=30,
        iterations=70,
    )
    summary_table(df, "osm_priority_zones")

    config = SimulationConfig(random_seed=42, priority_source="Real OSM Stirling zones")
    result = run_experiment(config=config, swarm_size=35, iterations=80)
    plot_comparison(
        config=result.config,
        users=result.users,
        baseline_drones=result.baseline_drones,
        optimized_drones=result.optimized_drones,
        baseline_graph=result.baseline_graph,
        optimized_graph=result.optimized_graph,
        output_path=OUTPUT_DIR / "osm_stirling_placement_example.png",
        priority_zones=result.priority_zones,
        kmeans_drones=result.kmeans_drones,
        kmeans_graph=result.kmeans_graph,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    default_df = run_many(
        name="default_30_seed",
        seeds=list(range(42, 72)),
        config_kwargs={},
        swarm_size=45,
        iterations=120,
    )
    default_summary = summary_table(default_df, "default_30_seed")
    default_result_figures(default_df, default_summary)

    run_uav_sensitivity()
    run_weight_sensitivity()
    run_osm_example()

    print(f"Evidence written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
