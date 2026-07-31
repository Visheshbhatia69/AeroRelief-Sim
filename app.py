"""Streamlit dashboard for the UAV disaster communication simulator."""

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from experiment import ExperimentResult, add_improvement_metrics, run_experiment
from metrics import evaluate_solution
from optimization import (
    OBJECTIVE_WEIGHTS,
    WEIGHT_PROFILES,
    ParticleSwarmOptimizer,
    kmeans_drone_placement,
)
from simulation import DisasterNetworkSimulation, SimulationConfig
from visualization import (
    plot_3d_deployment,
    plot_comparison,
    plot_metric_charts,
    plot_pso_convergence,
)


OUTPUT_DIR = Path("dashboard_outputs")
OBJECTIVE_TERM_COLORS = {
    "CU": "#7b2cbf",
    "PZ": "#c9184a",
    "AC": "#ffb703",
    "RL": "#6c757d",
    "MC": "#212529",
}


def main() -> None:
    """Run the dashboard."""
    st.set_page_config(
        page_title="AeroRelief-Sim: Priority-Aware UAV Placement",
        layout="wide",
    )
    apply_drone_theme()
    st.markdown(
        """
        <div class="uav-hero">
            <div>
                <div class="uav-kicker">AERORELIEF-SIM | DISASTER COMMUNICATION RELAY PLANNER</div>
                <h1>AeroRelief-Sim</h1>
                <p>A priority-aware UAV placement simulator for disaster communication.</p>
            </div>
            <div class="uav-hero-visual">
                <div class="uav-drone">
                    <span class="rotor rotor-a"></span>
                    <span class="rotor rotor-b"></span>
                    <span class="rotor rotor-c"></span>
                    <span class="rotor rotor-d"></span>
                    <span class="body"></span>
                </div>
                <div class="uav-signal">
                    <span></span><span></span><span></span><span></span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    config, swarm_size, iterations = sidebar_controls()
    show_flight_status(config, swarm_size, iterations)
    show_project_focus()

    if st.sidebar.button("Run deployment simulation", type="primary"):
        show_launch_sequence()
        with st.spinner("Placing baseline UAVs and optimizing UAV positions with PSO..."):
            result = run_experiment(config, swarm_size=swarm_size, iterations=iterations)
            st.session_state["result"] = result
            st.session_state["output_paths"] = save_dashboard_figures(result)

    if "result" not in st.session_state:
        st.info("Set the simulator controls in the sidebar, then run the deployment simulation.")
        return

    result = st.session_state["result"]
    output_paths = st.session_state["output_paths"]
    show_summary(result)
    show_mission_widgets(result)

    map_tab, view_3d_tab, metrics_tab, analysis_tab, findings_tab = st.tabs(
        [
            "Deployment map",
            "3D view",
            "Performance metrics",
            "Sensitivity analysis",
            "Major findings",
        ]
    )

    with map_tab:
        st.image(str(output_paths["comparison"]), use_container_width=True)
        show_figure_note(
            "Figure evidence",
            "The deployment map compares the spatial behaviour of each placement method. "
            "Random/static placement provides the reference case, K-Means follows user "
            "clusters, and PSO balances user connectivity with weighted priority-zone demand.",
        )
        st.subheader("UAV repositioning paths")
        st.image(str(output_paths["movement_paths"]), use_container_width=True)
        st.caption(
            "Arrows show repositioning from the random/static start to K-Means and PSO placements."
        )
        show_figure_note(
            "Movement interpretation",
            "The movement-path diagram links performance improvement to repositioning cost. "
            "This supports the energy-efficiency argument because larger gains should be "
            "considered together with the distance UAVs must move.",
        )

    with view_3d_tab:
        st.image(str(output_paths["deployment_3d"]), use_container_width=True)
        st.caption(
            "Users and priority zones are on the ground plane. UAVs are displayed at "
            "the configured altitude, coverage footprints are projected onto the ground, "
            "and optional obstacle blocks can reduce line-of-sight connectivity."
        )
        show_figure_note(
            "Model interpretation",
            "The 3D view demonstrates that the simulator is not only a flat 2D drawing. "
            "UAV altitude affects the ground coverage footprint, and obstacles can remove "
            "links even when a user is geographically close to a UAV.",
        )

    with metrics_tab:
        st.image(str(output_paths["metrics"]), use_container_width=True)
        show_figure_note(
            "Metric interpretation",
            "The main chart uses higher-is-better metrics only. Raw lower-is-better "
            "values such as connected components and movement cost remain in the table, "
            "while the graph converts fragmentation into a network cohesion score.",
        )
        show_metric_table(result)
        show_objective_breakdown(result)
        with st.expander("PSO convergence"):
            st.image(str(output_paths["convergence"]), use_container_width=True)
            st.caption("Shows whether the optimizer's best score improves over iterations.")
            show_figure_note(
                "AI behaviour",
                "The convergence curve shows whether PSO is learning better placements "
                "over iterations rather than returning a random-looking result.",
            )

    with analysis_tab:
        show_sensitivity_panel(config, swarm_size, iterations)

    with findings_tab:
        show_major_findings(result)
        show_evidence_summary(result)
        show_result_story_chart(result)
        show_objective_contribution_chart(result)
        show_demand_relief_heatmap(result)


def sidebar_controls() -> tuple[SimulationConfig, int, int]:
    """Collect simulator controls from the sidebar."""
    st.sidebar.header("Scenario")
    scenario_type = st.sidebar.selectbox(
        "User distribution",
        ["Random users", "Clustered users", "Hotspot priority zones"],
    )
    priority_source = st.sidebar.selectbox(
        "Priority-zone data",
        ["Synthetic demo zones", "Real OSM Stirling zones"],
    )

    st.sidebar.header("Map and nodes")
    area_size = st.sidebar.slider("Square area size (m)", 500, 3000, 1000, step=100)
    num_users = st.sidebar.slider("Users/rescue nodes", 20, 250, 80, step=10)
    num_drones = st.sidebar.slider("UAV relay nodes", 2, 12, 5)
    uav_altitude = st.sidebar.slider("UAV altitude (m)", 20, 300, 120, step=10)
    obstacles_enabled = st.sidebar.checkbox("Enable 3D obstacles", value=False)

    st.sidebar.header("Communication")
    user_range = st.sidebar.slider("User-to-UAV range (m)", 50, 500, 180, step=10)
    drone_range = st.sidebar.slider("UAV-to-UAV relay range (m)", 100, 1000, 350, step=25)

    with st.sidebar.expander("Optimizer settings"):
        iterations = st.slider("PSO iterations", 10, 250, 120, step=10)
        swarm_size = st.slider("PSO swarm size", 10, 100, 45, step=5)
        random_seed = st.number_input("Random seed", min_value=0, value=42, step=1)

    config = SimulationConfig(
        area_width=float(area_size),
        area_height=float(area_size),
        num_users=int(num_users),
        num_drones=int(num_drones),
        uav_altitude=float(uav_altitude),
        user_range=float(user_range),
        drone_range=float(drone_range),
        random_seed=int(random_seed),
        scenario_type=scenario_type,
        priority_source=priority_source,
        obstacles_enabled=obstacles_enabled,
    )
    return config, int(swarm_size), int(iterations)


def apply_drone_theme() -> None:
    """Apply a restrained UAV operations-console theme to the dashboard."""
    st.markdown(
        """
        <style>
        :root {
            --uav-bg: #071014;
            --uav-panel: #0e1b20;
            --uav-panel-soft: #13252b;
            --uav-border: #24424a;
            --uav-text: #e6f2f0;
            --uav-muted: #9db7b4;
            --uav-accent: #19c6a4;
            --uav-accent-2: #5ea4ff;
            --uav-warning: #ffb84d;
        }

        .stApp {
            background:
                linear-gradient(rgba(25, 198, 164, 0.035) 1px, transparent 1px),
                linear-gradient(90deg, rgba(25, 198, 164, 0.035) 1px, transparent 1px),
                radial-gradient(circle at 78% 12%, rgba(94, 164, 255, 0.14), transparent 24%),
                linear-gradient(135deg, #071014 0%, #0b161b 46%, #10191d 100%);
            background-size: 34px 34px, 34px 34px, auto, auto;
            color: var(--uav-text);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #081217 0%, #0d1a1f 100%);
            border-right: 1px solid var(--uav-border);
        }

        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: var(--uav-accent);
            letter-spacing: 0.02em;
        }

        .uav-hero {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 24px;
            padding: 22px 24px;
            margin-bottom: 18px;
            border: 1px solid var(--uav-border);
            border-radius: 8px;
            background: linear-gradient(135deg, rgba(14, 27, 32, 0.96), rgba(19, 37, 43, 0.88));
            box-shadow: 0 16px 36px rgba(0, 0, 0, 0.28);
        }

        .uav-hero h1 {
            color: var(--uav-text);
            font-size: 2.2rem;
            margin: 0.2rem 0 0.3rem 0;
            letter-spacing: 0;
        }

        .uav-hero p {
            color: var(--uav-muted);
            margin: 0;
            font-size: 1.02rem;
        }

        .uav-kicker {
            color: var(--uav-accent);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
        }

        .uav-hero-visual {
            display: flex;
            align-items: center;
            gap: 20px;
        }

        .uav-signal {
            display: flex;
            align-items: flex-end;
            gap: 5px;
            min-width: 64px;
            height: 44px;
        }

        .uav-signal span {
            display: block;
            width: 9px;
            border-radius: 3px 3px 0 0;
            background: var(--uav-accent);
            box-shadow: 0 0 16px rgba(25, 198, 164, 0.65);
        }

        .uav-signal span:nth-child(1) { height: 14px; opacity: 0.45; }
        .uav-signal span:nth-child(2) { height: 22px; opacity: 0.65; }
        .uav-signal span:nth-child(3) { height: 32px; opacity: 0.85; }
        .uav-signal span:nth-child(4) { height: 42px; opacity: 1; }

        .uav-drone {
            position: relative;
            width: 82px;
            height: 46px;
            animation: uav-hover 3.4s ease-in-out infinite;
            filter: drop-shadow(0 0 14px rgba(25, 198, 164, 0.42));
        }

        .uav-drone .body {
            position: absolute;
            left: 31px;
            top: 18px;
            width: 20px;
            height: 12px;
            border-radius: 6px;
            background: var(--uav-accent);
            box-shadow: 0 0 18px rgba(25, 198, 164, 0.5);
        }

        .uav-drone::before,
        .uav-drone::after {
            content: "";
            position: absolute;
            left: 12px;
            right: 12px;
            top: 22px;
            height: 2px;
            background: rgba(230, 242, 240, 0.78);
        }

        .uav-drone::after {
            transform: rotate(90deg);
        }

        .rotor {
            position: absolute;
            width: 22px;
            height: 22px;
            border: 2px solid rgba(94, 164, 255, 0.86);
            border-radius: 50%;
            animation: rotor-pulse 0.55s linear infinite;
        }

        .rotor-a { left: 0; top: 0; }
        .rotor-b { right: 0; top: 0; }
        .rotor-c { left: 0; bottom: 0; }
        .rotor-d { right: 0; bottom: 0; }

        .uav-launch {
            position: relative;
            overflow: hidden;
            min-height: 74px;
            margin: 8px 0 14px 0;
            padding: 14px 16px;
            border: 1px solid rgba(25, 198, 164, 0.35);
            border-radius: 8px;
            background: linear-gradient(135deg, rgba(14, 27, 32, 0.92), rgba(19, 37, 43, 0.72));
        }

        .uav-launch .track {
            position: absolute;
            left: 18px;
            right: 18px;
            top: 48px;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(94, 164, 255, 0.72), transparent);
        }

        .uav-launch .mini-drone {
            position: absolute;
            top: 35px;
            left: 22px;
            width: 18px;
            height: 8px;
            border-radius: 5px;
            background: var(--uav-accent);
            animation: scan-flight 2.2s ease-in-out infinite;
            box-shadow: 0 0 14px rgba(25, 198, 164, 0.65);
        }

        .uav-launch .mini-drone::before,
        .uav-launch .mini-drone::after {
            content: "";
            position: absolute;
            top: -5px;
            width: 8px;
            height: 8px;
            border: 1px solid rgba(94, 164, 255, 0.9);
            border-radius: 50%;
        }

        .uav-launch .mini-drone::before { left: -9px; }
        .uav-launch .mini-drone::after { right: -9px; }

        .uav-launch strong {
            color: var(--uav-accent);
            letter-spacing: 0.03em;
        }

        .uav-widget-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin: 16px 0 18px 0;
        }

        .uav-widget {
            min-height: 92px;
            padding: 14px 16px;
            border: 1px solid rgba(36, 66, 74, 0.9);
            border-radius: 8px;
            background: linear-gradient(180deg, rgba(14, 27, 32, 0.88), rgba(9, 19, 23, 0.82));
            position: relative;
            overflow: hidden;
        }

        .uav-widget::after {
            content: "";
            position: absolute;
            left: -30%;
            right: -30%;
            bottom: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(25, 198, 164, 0.8), transparent);
            animation: widget-sweep 4.2s linear infinite;
        }

        .uav-widget .label {
            color: var(--uav-muted);
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 8px;
        }

        .uav-widget .value {
            color: var(--uav-text);
            font-size: 1.15rem;
            font-weight: 700;
        }

        .uav-widget .note {
            color: var(--uav-muted);
            font-size: 0.82rem;
            margin-top: 7px;
        }

        .figure-note {
            margin: 10px 0 18px 0;
            padding: 12px 14px;
            border-left: 3px solid var(--uav-accent-2);
            border-radius: 6px;
            background: rgba(14, 27, 32, 0.72);
            color: var(--uav-muted);
            font-size: 0.95rem;
            line-height: 1.45;
        }

        .figure-note-title {
            color: var(--uav-accent-2);
            font-weight: 700;
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            font-size: 0.74rem;
        }

        @keyframes uav-hover {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-7px); }
        }

        @keyframes rotor-pulse {
            0% { transform: scale(0.92); opacity: 0.55; }
            50% { transform: scale(1.08); opacity: 1; }
            100% { transform: scale(0.92); opacity: 0.55; }
        }

        @keyframes scan-flight {
            0% { left: 24px; transform: translateY(0); }
            48% { transform: translateY(-8px); }
            100% { left: calc(100% - 48px); transform: translateY(0); }
        }

        @keyframes widget-sweep {
            0% { transform: translateX(-45%); opacity: 0; }
            20% { opacity: 1; }
            100% { transform: translateX(45%); opacity: 0; }
        }

        .uav-status-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 10px;
            margin-bottom: 16px;
        }

        .uav-status-card {
            padding: 12px 14px;
            border: 1px solid var(--uav-border);
            border-radius: 8px;
            background: rgba(14, 27, 32, 0.82);
        }

        .uav-status-card .label {
            color: var(--uav-muted);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }

        .uav-status-card .value {
            color: var(--uav-text);
            font-size: 1.08rem;
            font-weight: 700;
        }

        div[data-testid="stMetric"] {
            padding: 12px 14px;
            border: 1px solid rgba(36, 66, 74, 0.9);
            border-radius: 8px;
            background: rgba(14, 27, 32, 0.78);
        }

        div[data-testid="stTabs"] button {
            color: var(--uav-muted);
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--uav-accent);
            border-bottom-color: var(--uav-accent);
        }

        div[data-testid="stAlert"] {
            border-radius: 8px;
            border: 1px solid rgba(36, 66, 74, 0.75);
        }

        @media (max-width: 900px) {
            .uav-hero {
                align-items: flex-start;
                flex-direction: column;
            }
            .uav-status-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .uav-widget-grid {
                grid-template-columns: repeat(1, minmax(0, 1fr));
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_flight_status(config: SimulationConfig, swarm_size: int, iterations: int) -> None:
    """Show a compact UAV mission-status strip."""
    obstacle_mode = "Obstacle-aware" if config.obstacles_enabled else "Clear line-of-sight"
    status_items = [
        ("UAV fleet", f"{config.num_drones} relay nodes"),
        ("Altitude", f"{config.uav_altitude:.0f} m"),
        ("Area", f"{config.area_width:.0f} x {config.area_height:.0f} m"),
        ("Link range", f"{config.user_range:.0f} m"),
        ("Mode", obstacle_mode),
    ]
    cards = "".join(
        (
            '<div class="uav-status-card">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{value}</div>'
            "</div>"
        )
        for label, value in status_items
    )
    st.html(f'<div class="uav-status-grid">{cards}</div>')


def show_figure_note(title: str, text: str) -> None:
    """Render a compact interpretation note under an important figure."""
    st.markdown(
        f"""
        <div class="figure-note">
            <div class="figure-note-title">{title}</div>
            <div>{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_launch_sequence() -> None:
    """Show a small animated launch panel while the simulation is running."""
    st.html(
        """
        <div class="uav-launch">
            <strong>MISSION RUNNING</strong>
            <div>Deploying baseline UAVs, optimizing PSO placement, and preparing result maps.</div>
            <div class="track"></div>
            <div class="mini-drone"></div>
        </div>
        """
    )


def show_mission_widgets(result: ExperimentResult) -> None:
    """Show a professional summary strip after a simulation run."""
    baseline = result.baseline_metrics
    optimized = result.optimized_metrics
    connected_gain = optimized["connected_users_percent"] - baseline["connected_users_percent"]
    priority_gain = optimized["priority_coverage_percent"] - baseline["priority_coverage_percent"]
    fragment_drop = baseline["connected_components"] - optimized["connected_components"]

    items = [
        (
            "Mission outcome",
            f"+{connected_gain:.1f} pp users",
            "PSO gain over random/static placement",
        ),
        (
            "Priority response",
            f"+{priority_gain:.1f} pp zones",
            "Weighted emergency-location coverage gain",
        ),
        (
            "Network condition",
            f"{fragment_drop:.0f} fewer fragments",
            "Lower fragmentation means a less broken network",
        ),
        (
            "AI search score",
            f"{result.best_score:.3f}",
            "Best normalized PSO objective value",
        ),
    ]
    cards = "".join(
        (
            '<div class="uav-widget">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{value}</div>'
            f'<div class="note">{note}</div>'
            "</div>"
        )
        for label, value, note in items
    )
    st.html(f'<div class="uav-widget-grid">{cards}</div>')


def show_project_focus() -> None:
    """Show a short explanation of what the simulator is doing."""
    col1, col2, col3 = st.columns(3)
    col1.info("**Simulator goal**\n\nPlace UAV relay nodes to improve disaster communication.")
    col2.info("**Comparison**\n\nRandom/static and K-Means placement are used as baselines.")
    col3.info("**AI method**\n\nPSO searches for better UAV positions using the weighted objective.")

    with st.expander("Objective formula"):
        st.code("Score = 0.40(CU) + 0.35(PZ) + 0.10(AC) + 0.10(RL) - 0.05(MC)")
        st.markdown(
            """
            **CU** connected-user ratio, **PZ** priority-zone coverage,
            **AC** area coverage, **RL** UAV relay-link quality,
            **MC** movement cost. This priority-aware weighting matches the paper topic,
            and the absolute weights add up to 1.
            """
        )


def save_dashboard_figures(result: ExperimentResult) -> dict[str, Path]:
    """Save dashboard figures and return their paths."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    paths = {
        "comparison": OUTPUT_DIR / "dashboard_comparison.png",
        "deployment_3d": OUTPUT_DIR / "dashboard_3d_deployment.png",
        "movement_paths": OUTPUT_DIR / "dashboard_movement_paths.png",
        "metrics": OUTPUT_DIR / "dashboard_metrics.png",
        "convergence": OUTPUT_DIR / "dashboard_pso_convergence.png",
    }

    plot_comparison(
        config=result.config,
        users=result.users,
        baseline_drones=result.baseline_drones,
        optimized_drones=result.optimized_drones,
        baseline_graph=result.baseline_graph,
        optimized_graph=result.optimized_graph,
        output_path=paths["comparison"],
        priority_zones=result.priority_zones,
        kmeans_drones=result.kmeans_drones,
        kmeans_graph=result.kmeans_graph,
    )
    plot_metric_charts(
        result.baseline_metrics,
        result.optimized_metrics,
        paths["metrics"],
        kmeans_metrics=result.kmeans_metrics,
    )
    plot_3d_deployment(
        config=result.config,
        users=result.users,
        baseline_drones=result.baseline_drones,
        kmeans_drones=result.kmeans_drones,
        optimized_drones=result.optimized_drones,
        output_path=paths["deployment_3d"],
        priority_zones=result.priority_zones,
    )
    plot_pso_convergence(result.pso_score_history, paths["convergence"])
    plot_uav_movement_paths(result, paths["movement_paths"])
    return paths


def plot_uav_movement_paths(result: ExperimentResult, output_path: Path) -> None:
    """Show how each UAV moves from random baseline to K-Means and PSO positions."""
    fig, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
    config = result.config
    axis.set_title("UAV repositioning paths")
    axis.set_xlim(0.0, config.area_width)
    axis.set_ylim(0.0, config.area_height)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x position (m)")
    axis.set_ylabel("y position (m)")
    axis.grid(True, linestyle=":", alpha=0.35)

    axis.scatter(
        result.users[:, 0],
        result.users[:, 1],
        s=14,
        c="#4c78a8",
        alpha=0.35,
        label="Users",
    )
    axis.scatter(
        result.baseline_drones[:, 0],
        result.baseline_drones[:, 1],
        s=120,
        c="#d95f02",
        marker="^",
        edgecolors="black",
        label="Random/static start",
    )
    axis.scatter(
        result.kmeans_drones[:, 0],
        result.kmeans_drones[:, 1],
        s=120,
        c="#4c78a8",
        marker="^",
        edgecolors="black",
        label="K-Means placement",
    )
    axis.scatter(
        result.optimized_drones[:, 0],
        result.optimized_drones[:, 1],
        s=130,
        c="#1b9e77",
        marker="^",
        edgecolors="black",
        label="PSO placement",
    )

    for index, (start, kmeans, pso) in enumerate(
        zip(result.baseline_drones, result.kmeans_drones, result.optimized_drones)
    ):
        axis.annotate(
            "",
            xy=kmeans,
            xytext=start,
            arrowprops={"arrowstyle": "->", "color": "#4c78a8", "lw": 1.5, "alpha": 0.75},
        )
        axis.annotate(
            "",
            xy=pso,
            xytext=start,
            arrowprops={"arrowstyle": "->", "color": "#1b9e77", "lw": 1.8, "alpha": 0.85},
        )
        axis.text(start[0] + 8, start[1] + 8, f"D{index}", fontsize=8, weight="bold")

    axis.legend(loc="upper right", fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def show_summary(result: ExperimentResult) -> None:
    """Show top-level result metrics."""
    baseline = result.baseline_metrics
    optimized = result.optimized_metrics

    st.subheader("Deployment result")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Connected users",
        f"{optimized['connected_users_percent']:.1f}%",
        f"{optimized['connected_users_gain_pp']:.1f} pp",
    )
    col2.metric(
        "Priority coverage",
        f"{optimized['priority_coverage_percent']:.1f}%",
        f"{optimized['priority_coverage_gain_pp']:.1f} pp",
    )
    col3.metric(
        "Network fragments",
        f"{optimized['connected_components']:.0f}",
        f"{optimized['connected_components'] - baseline['connected_components']:.0f}",
        delta_color="inverse",
    )
    col4.metric("Movement cost", f"{optimized['movement_cost']:.0f} m")


def show_major_findings(result: ExperimentResult) -> None:
    """Summarize the main technical findings from one simulation run."""
    baseline = result.baseline_metrics
    kmeans = result.kmeans_metrics
    optimized = result.optimized_metrics

    connected_gain = optimized["connected_users_percent"] - baseline["connected_users_percent"]
    priority_gain = optimized["priority_coverage_percent"] - baseline["priority_coverage_percent"]
    fragment_drop = baseline["connected_components"] - optimized["connected_components"]
    kmeans_priority_gap = optimized["priority_coverage_percent"] - kmeans["priority_coverage_percent"]

    st.subheader("Main result interpretation")
    col1, col2, col3 = st.columns(3)
    col1.success(
        f"**PSO improves user connectivity**\n\n"
        f"Connected users increased by **{connected_gain:.1f} percentage points** "
        f"compared with random/static placement."
    )
    col2.success(
        f"**Priority-aware placement matters**\n\n"
        f"Priority-zone coverage improved by **{priority_gain:.1f} percentage points** "
        f"over random/static placement."
    )
    col3.info(
        f"**Network fragmentation reduces**\n\n"
        f"The number of disconnected network groups reduced by **{fragment_drop:.0f}**, "
        f"so the network is less broken up."
    )

    st.markdown(
        f"""
        The optimized placement is stronger because it does not only follow user
        clusters. It also rewards weighted priority-zone coverage, UAV relay links,
        and movement cost. In this run, PSO is **{kmeans_priority_gap:.1f} percentage
        points** better than K-Means for priority-zone coverage.
        """
    )


def show_evidence_summary(result: ExperimentResult) -> None:
    """Summarize what each result view contributes to the dissertation evidence."""
    rows = [
        {
            "Evidence question": "Does AI placement improve service?",
            "Dashboard evidence": "Slope chart and metric table",
            "Interpretation": "Compares random/static, K-Means, and PSO on connected users and priority coverage.",
        },
        {
            "Evidence question": "Why is the PSO result better?",
            "Dashboard evidence": "Objective contribution chart",
            "Interpretation": "Shows which weighted terms increase or penalise each placement.",
        },
        {
            "Evidence question": "Where does the network remain weak?",
            "Dashboard evidence": "Before/after demand heatmap",
            "Interpretation": "Shows demand concentration before PSO and remaining uncovered demand afterwards.",
        },
        {
            "Evidence question": "Are the results robust?",
            "Dashboard evidence": "Sensitivity analysis",
            "Interpretation": "Tests UAV count, range, area size, iterations, scenarios, repeated runs, and weights.",
        },
    ]
    st.subheader("Evidence summary")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def show_result_story_chart(result: ExperimentResult) -> None:
    """Show before/after improvement using a slope chart instead of another bar chart."""
    methods = ["Random/static", "K-Means", "PSO"]
    connected = [
        result.baseline_metrics["connected_users_percent"],
        result.kmeans_metrics["connected_users_percent"],
        result.optimized_metrics["connected_users_percent"],
    ]
    priority = [
        result.baseline_metrics["priority_coverage_percent"],
        result.kmeans_metrics["priority_coverage_percent"],
        result.optimized_metrics["priority_coverage_percent"],
    ]
    fragments = [
        result.baseline_metrics["connected_components"],
        result.kmeans_metrics["connected_components"],
        result.optimized_metrics["connected_components"],
    ]
    cohesion = relative_cohesion_scores(fragments)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)

    for values, label, color in [
        (connected, "Connected users (%)", "#4c78a8"),
        (priority, "Priority coverage (%)", "#9467bd"),
    ]:
        axes[0].plot(methods, values, marker="o", linewidth=2.5, label=label, color=color)
        for index, value in enumerate(values):
            axes[0].text(index, value + 2, f"{value:.1f}", ha="center", fontsize=9)
    axes[0].set_title("Placement method improvement")
    axes[0].set_ylabel("Performance (%)")
    axes[0].set_ylim(0, max(105, max(connected + priority) + 10))
    axes[0].grid(axis="y", linestyle=":", alpha=0.35)
    axes[0].legend()

    axes[1].plot(methods, cohesion, marker="o", linewidth=2.5, color="#d95f02")
    for index, value in enumerate(cohesion):
        axes[1].text(index, value + 2, f"{value:.1f}", ha="center", fontsize=9)
    axes[1].set_title("Network cohesion score")
    axes[1].set_ylabel("Relative score (%)")
    axes[1].set_ylim(0, 110)
    axes[1].grid(axis="y", linestyle=":", alpha=0.35)

    fig.suptitle("Major finding: higher scores indicate stronger disaster communication support")
    st.pyplot(fig)
    plt.close(fig)


def relative_cohesion_scores(component_counts: list[float]) -> list[float]:
    """Convert lower-is-better component counts into higher-is-better scores."""
    values = np.array(component_counts, dtype=float)
    worst = float(values.max())
    best = float(values.min())
    if worst == best:
        return [100.0 for _ in values]
    return [float(100.0 * (worst - value) / (worst - best)) for value in values]


def show_objective_contribution_chart(result: ExperimentResult) -> None:
    """Visualize how each objective-function term contributes to each method."""
    rows = pd.DataFrame(
        [
            *objective_rows("Random/static", result, result.baseline_drones, result.baseline_graph),
            *objective_rows("K-Means", result, result.kmeans_drones, result.kmeans_graph),
            *objective_rows("PSO", result, result.optimized_drones, result.optimized_graph),
        ]
    )
    pivot = rows.pivot_table(
        index="Placement",
        columns="Term",
        values="Weighted contribution",
        aggfunc="sum",
    ).reindex(["Random/static", "K-Means", "PSO"])

    terms = ["CU", "PZ", "AC", "RL", "MC"]
    colors = OBJECTIVE_TERM_COLORS

    fig, axis = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    positive_bottom = np.zeros(len(pivot))
    negative_bottom = np.zeros(len(pivot))
    x_values = np.arange(len(pivot))

    for term in terms:
        values = pivot[term].fillna(0).to_numpy()
        bottom = np.where(values >= 0, positive_bottom, negative_bottom)
        axis.bar(x_values, values, bottom=bottom, label=term, color=colors[term])
        positive_bottom += np.where(values >= 0, values, 0)
        negative_bottom += np.where(values < 0, values, 0)

    totals = pivot[terms].sum(axis=1)
    for index, total in enumerate(totals):
        axis.text(index, total + 0.015, f"{total:.3f}", ha="center", fontsize=9)

    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.set_xticks(x_values)
    axis.set_xticklabels(pivot.index)
    axis.set_ylabel("Weighted objective contribution")
    axis.set_title("Why the optimized placement scores better")
    axis.grid(axis="y", linestyle=":", alpha=0.35)
    axis.legend(title="Objective term", ncol=5, fontsize=8)
    axis.text(
        0.01,
        -0.18,
        "Objective-term colours are separate from placement-method colours.",
        transform=axis.transAxes,
        fontsize=8,
        color="#555555",
    )
    st.pyplot(fig)
    plt.close(fig)


def show_demand_relief_heatmap(result: ExperimentResult) -> None:
    """Show demand concentration before and remaining need after PSO coverage."""
    st.subheader("Communication demand heatmap: before vs after PSO")
    st.caption(
        "Before shows where communication demand is concentrated from users and "
        "priority zones. After shows the demand still not covered by the PSO UAV "
        "deployment. This is an explanatory map, not a separate prediction model."
    )

    before_grid, after_grid, x_values, y_values, diagnostics = demand_relief_grids(result)
    config = result.config
    remaining_percent = diagnostics["remaining_demand_percent"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), constrained_layout=True)
    vmax = max(float(before_grid.max()), 1e-9)
    plot_specs = [
        (axes[0], before_grid, "Before optimization: communication demand pressure"),
        (axes[1], after_grid, "After PSO: remaining uncovered demand"),
    ]

    for axis, grid, title in plot_specs:
        image = axis.imshow(
            grid,
            origin="lower",
            extent=[0, config.area_width, 0, config.area_height],
            cmap="YlOrRd",
            vmin=0.0,
            vmax=vmax,
            alpha=0.82,
            aspect="equal",
        )
        axis.scatter(
            result.users[:, 0],
            result.users[:, 1],
            s=10,
            c="#1f4e79",
            alpha=0.38,
            label="Users",
        )
        axis.set_title(title)
        axis.set_xlabel("x position (m)")
        axis.set_ylabel("y position (m)")
        axis.grid(color="white", alpha=0.12, linewidth=0.5)

    fig.colorbar(image, ax=axes, label="Relative communication need")

    axes[1].scatter(
        result.optimized_drones[:, 0],
        result.optimized_drones[:, 1],
        s=130,
        c="#19c6a4",
        marker="^",
        edgecolors="black",
        label="PSO UAVs",
        zorder=5,
    )
    covered_zone_indices = covered_priority_zone_indices(result, result.optimized_drones)
    for axis in axes:
        for zone_index, zone in enumerate(result.priority_zones):
            is_covered = zone_index in covered_zone_indices
            circle = plt.Circle(
                zone.center,
                zone.radius,
                fill=False,
                edgecolor="#6a1b9a" if not is_covered else "#555555",
                linewidth=1.8 if not is_covered else 1.0,
                alpha=0.9 if not is_covered else 0.45,
            )
            axis.add_patch(circle)
            label = zone.name.split(" (")[0]
            if axis is axes[1] and not is_covered:
                label = f"{label} gap"
            axis.text(zone.center[0], zone.center[1], label, fontsize=7)
        axis.legend(loc="upper right", fontsize=8)

    fig.suptitle("Demand relief: how much communication need the PSO UAVs remove")
    st.pyplot(fig)
    plt.close(fig)

    col1, col2, col3 = st.columns(3)
    col1.metric("Demand relieved", f"{100.0 - remaining_percent:.1f}%")
    col2.metric("Demand still visible", f"{remaining_percent:.1f}%")
    col3.metric("Basis", "Users + weighted zones")


def demand_relief_grids(
    result: ExperimentResult,
    grid_size: int = 55,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Create before/after heatmaps from users, weighted zones, and UAV coverage."""
    config = result.config
    simulation = DisasterNetworkSimulation(config)
    obstacles = simulation.generate_obstacles()

    x_values = np.linspace(0.0, config.area_width, grid_size)
    y_values = np.linspace(0.0, config.area_height, grid_size)
    xx, yy = np.meshgrid(x_values, y_values)
    points = np.column_stack((xx.ravel(), yy.ravel()))

    user_sigma = max(config.user_range * 0.45, 35.0)
    zone_sigma = max(config.user_range * 0.65, 60.0)
    demand = np.zeros(len(points), dtype=float)

    for user_position in result.users:
        squared_distance = np.sum((points - user_position) ** 2, axis=1)
        demand += np.exp(-squared_distance / (2.0 * user_sigma**2))

    for zone in result.priority_zones:
        zone_center = np.array(zone.center)
        squared_distance = np.sum((points - zone_center) ** 2, axis=1)
        demand += zone.weight * 2.0 * np.exp(-squared_distance / (2.0 * zone_sigma**2))

    served = np.zeros(len(points), dtype=bool)
    ground_radius = simulation.ground_coverage_radius()
    for point_index, point in enumerate(points):
        for drone in result.optimized_drones:
            distance = np.linalg.norm(point - drone)
            if distance <= ground_radius and not simulation.is_link_blocked(
                point, drone, obstacles
            ):
                served[point_index] = True
                break

    before_grid = demand.reshape(grid_size, grid_size)
    after_grid = np.where(served, demand * 0.15, demand).reshape(grid_size, grid_size)
    max_value = before_grid.max()
    if max_value > 0:
        before_grid = before_grid / max_value
        after_grid = after_grid / max_value

    total_before = float(before_grid.sum())
    total_after = float(after_grid.sum())
    diagnostics = {
        "remaining_demand_percent": 100.0 * total_after / max(total_before, 1e-9),
    }
    return before_grid, after_grid, x_values, y_values, diagnostics


def covered_priority_zone_indices(
    result: ExperimentResult,
    drones: np.ndarray,
) -> set[int]:
    """Return priority zones already covered by a UAV placement."""
    simulation = DisasterNetworkSimulation(result.config)
    ground_radius = simulation.ground_coverage_radius()
    obstacles = simulation.generate_obstacles()
    covered = set()
    for zone_index, zone in enumerate(result.priority_zones):
        zone_center = np.array(zone.center)
        distances = np.linalg.norm(drones - zone_center, axis=1)
        for drone, distance in zip(drones, distances):
            if distance <= ground_radius + zone.radius and not simulation.is_link_blocked(
                zone_center, drone, obstacles
            ):
                covered.add(zone_index)
                break
    return covered


def show_metric_table(result: ExperimentResult) -> None:
    """Show metric values in a compact table."""
    methods = {
        "Random/static": result.baseline_metrics,
        "K-Means": result.kmeans_metrics,
        "PSO optimized": result.optimized_metrics,
    }
    rows = [
        metric_row("Connected users", methods, "connected_users_percent", "%"),
        metric_row("Area coverage", methods, "coverage_percent", "%"),
        metric_row("Priority-zone coverage", methods, "priority_coverage_percent", "weighted %"),
        metric_row("Network fragments", methods, "connected_components", "count"),
        metric_row("Average path length", methods, "average_path_length", "hops"),
        metric_row("Movement cost", methods, "movement_cost", "metres"),
    ]
    st.table(rows)


def show_objective_breakdown(result: ExperimentResult) -> None:
    """Show how the PSO objective score is calculated."""
    st.subheader("Objective score calculation")
    st.caption(
        "This table shows the exact weighted terms used by PSO to judge each UAV placement."
    )

    df = pd.DataFrame(
        [
            *objective_rows("Baseline", result, result.baseline_drones, result.baseline_graph),
            *objective_rows("K-Means", result, result.kmeans_drones, result.kmeans_graph),
            *objective_rows("PSO optimized", result, result.optimized_drones, result.optimized_graph),
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    score_summary = (
        df.groupby("Placement", as_index=False)["Weighted contribution"]
        .sum()
        .rename(columns={"Weighted contribution": "Objective score"})
    )
    st.table(score_summary.round(4))


def objective_rows(
    placement: str,
    result: ExperimentResult,
    drones,
    graph,
) -> list[dict[str, object]]:
    """Return objective-score calculation rows for one UAV placement."""
    config = result.config
    metrics = (
        result.baseline_metrics
        if placement in {"Baseline", "Random/static"}
        else result.kmeans_metrics
        if placement == "K-Means"
        else result.optimized_metrics
    )
    connected_ratio = metrics["connected_users_percent"] / 100.0
    priority_ratio = metrics["priority_coverage_percent"] / 100.0
    area_coverage_ratio = metrics["coverage_percent"] / 100.0

    drone_nodes = [f"drone_{index}" for index in range(config.num_drones)]
    relay_edges = graph.subgraph(drone_nodes).number_of_edges()
    max_relay_edges = max(1, config.num_drones * (config.num_drones - 1) / 2)
    relay_ratio = relay_edges / max_relay_edges

    movement_cost = float(np.sum(np.linalg.norm(drones - result.baseline_drones, axis=1)))
    normalized_movement = movement_cost / (
        config.num_drones * np.hypot(config.area_width, config.area_height)
    )

    terms = [
        ("CU", "Connected-user ratio", OBJECTIVE_WEIGHTS["connected_users"], connected_ratio),
        ("PZ", "Priority-zone coverage", OBJECTIVE_WEIGHTS["priority_coverage"], priority_ratio),
        ("AC", "Area coverage", OBJECTIVE_WEIGHTS["area_coverage"], area_coverage_ratio),
        ("RL", "UAV relay-link ratio", OBJECTIVE_WEIGHTS["relay_links"], relay_ratio),
        ("MC", "Movement cost penalty", -OBJECTIVE_WEIGHTS["movement_cost"], normalized_movement),
    ]

    rows = []
    for symbol, meaning, weight, value in terms:
        rows.append(
            {
                "Placement": placement,
                "Term": symbol,
                "Meaning": meaning,
                "Raw value": round(value, 4),
                "Weight": weight,
                "Weighted contribution": round(weight * value, 4),
            }
        )
    return rows


def metric_row(
    label: str,
    methods: dict[str, dict[str, float]],
    key: str,
    unit: str,
) -> dict[str, str]:
    """Format one metric-table row."""
    row = {
        "Metric": label,
        "Unit": unit,
    }
    for method_name, values in methods.items():
        row[method_name] = format_metric_value(values[key], unit)
    return row


def format_metric_value(value: float, unit: str) -> str:
    """Format values cleanly for the dashboard table."""
    if unit in {"%", "weighted %"}:
        return f"{value:.2f}%"
    if unit in {"count", "metres"}:
        return f"{value:.0f}"
    return f"{value:.2f}"


def show_sensitivity_panel(
    config: SimulationConfig,
    swarm_size: int,
    iterations: int,
) -> None:
    """Let the user run one clear sensitivity analysis at a time."""
    st.markdown(
        "Use this to test one question at a time, such as whether adding UAVs or "
        "increasing communication range improves the deployment."
    )
    analysis_type = st.selectbox(
        "Analysis to run",
        [
            "Number of UAVs",
            "Disaster area size",
            "Communication range",
            "PSO iterations",
            "User distribution scenario",
            "Repeated runs reliability",
            "Dynamic user motion",
            "Objective weight sensitivity",
        ],
    )

    if st.button("Run selected analysis"):
        with st.spinner(f"Running analysis: {analysis_type}..."):
            st.session_state["analysis_type"] = analysis_type
            st.session_state["analysis_df"] = run_selected_analysis(
                analysis_type, config, swarm_size, iterations
            )

    if "analysis_df" not in st.session_state:
        st.info("Select an analysis and run it to generate a graph and CSV results.")
        return

    df = st.session_state["analysis_df"]
    current_type = st.session_state["analysis_type"]
    st.dataframe(df, use_container_width=True)
    show_selected_analysis_chart(current_type, df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download analysis CSV",
        data=csv,
        file_name=f"{current_type.lower().replace(' ', '_')}_results.csv",
        mime="text/csv",
    )


def run_selected_analysis(
    analysis_type: str,
    config: SimulationConfig,
    swarm_size: int,
    iterations: int,
) -> pd.DataFrame:
    """Run the chosen sensitivity analysis."""
    if analysis_type == "Number of UAVs":
        return run_uav_sweep(config, swarm_size, iterations)
    if analysis_type == "Disaster area size":
        return run_area_sweep(config, swarm_size, iterations)
    if analysis_type == "Communication range":
        return run_range_sweep(config, swarm_size, iterations)
    if analysis_type == "PSO iterations":
        return run_iteration_sweep(config, swarm_size)
    if analysis_type == "User distribution scenario":
        return run_scenario_comparison(config, swarm_size, iterations)
    if analysis_type == "Dynamic user motion":
        return run_dynamic_motion_analysis(config, swarm_size, iterations)
    if analysis_type == "Objective weight sensitivity":
        return run_weight_sensitivity_analysis(config, swarm_size, iterations)
    return run_repeated_experiments(config, swarm_size, iterations, runs=10)


def show_selected_analysis_chart(analysis_type: str, df: pd.DataFrame) -> None:
    """Display the right chart for the chosen analysis."""
    if analysis_type == "Number of UAVs":
        show_sensitivity_plot(df, "uavs", "Number of UAVs", "Effect of UAV count")
    elif analysis_type == "Disaster area size":
        show_sensitivity_plot(df, "area_size_m", "Area size (m)", "Effect of area size")
    elif analysis_type == "Communication range":
        show_sensitivity_plot(
            df, "user_range_m", "User-to-UAV range (m)", "Effect of communication range"
        )
    elif analysis_type == "PSO iterations":
        show_sensitivity_plot(df, "iterations", "PSO iterations", "Effect of PSO iterations")
    elif analysis_type == "User distribution scenario":
        show_scenario_chart(df)
    elif analysis_type == "Dynamic user motion":
        show_dynamic_motion_chart(df)
    elif analysis_type == "Objective weight sensitivity":
        show_weight_sensitivity_chart(df)
    else:
        show_reliability_chart(df)


def run_repeated_experiments(
    config: SimulationConfig, swarm_size: int, iterations: int, runs: int
) -> pd.DataFrame:
    """Run the same scenario with different random seeds."""
    rows = []

    for index in range(runs):
        run_config = replace_config(config, random_seed=config.random_seed + index)
        result = run_experiment(run_config, swarm_size=swarm_size, iterations=iterations)
        rows.append(flatten_result(index + 1, result, config.scenario_type))

    return pd.DataFrame(rows)


def flatten_result(run_number: int, result: ExperimentResult, scenario_type: str) -> dict:
    """Convert one experiment result to one CSV/table row."""
    baseline = result.baseline_metrics
    kmeans = result.kmeans_metrics
    optimized = result.optimized_metrics
    return {
        "run": run_number,
        "scenario": scenario_type,
        "priority_source": result.config.priority_source,
        "users": result.config.num_users,
        "uavs": result.config.num_drones,
        "baseline_connected_users_percent": baseline["connected_users_percent"],
        "kmeans_connected_users_percent": kmeans["connected_users_percent"],
        "pso_connected_users_percent": optimized["connected_users_percent"],
        "kmeans_connected_users_gain_pp": kmeans["connected_users_gain_pp"],
        "connected_users_gain_pp": optimized["connected_users_gain_pp"],
        "baseline_priority_coverage_percent": baseline["priority_coverage_percent"],
        "kmeans_priority_coverage_percent": kmeans["priority_coverage_percent"],
        "pso_priority_coverage_percent": optimized["priority_coverage_percent"],
        "kmeans_priority_coverage_gain_pp": kmeans["priority_coverage_gain_pp"],
        "priority_coverage_gain_pp": optimized["priority_coverage_gain_pp"],
        "baseline_network_fragments": baseline["connected_components"],
        "kmeans_network_fragments": kmeans["connected_components"],
        "pso_network_fragments": optimized["connected_components"],
        "kmeans_movement_cost_m": kmeans["movement_cost"],
        "pso_movement_cost_m": optimized["movement_cost"],
        "best_pso_score": result.best_score,
    }


def show_sensitivity_plot(
    df: pd.DataFrame, x_column: str, x_label: str, title: str
) -> None:
    """Plot baseline and PSO performance for one changed parameter."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    axes[0].plot(
        df[x_column],
        df["baseline_connected_users_percent"],
        marker="o",
        label="Baseline",
        color="#d95f02",
    )
    if "kmeans_connected_users_percent" in df.columns:
        axes[0].plot(
            df[x_column],
            df["kmeans_connected_users_percent"],
            marker="o",
            label="K-Means",
            color="#4c78a8",
        )
    axes[0].plot(
        df[x_column],
        df["pso_connected_users_percent"],
        marker="o",
        label="PSO",
        color="#1b9e77",
    )
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel("Connected users (%)")
    axes[0].set_title("User connectivity")
    axes[0].grid(True, linestyle=":", alpha=0.35)
    axes[0].legend()

    axes[1].plot(
        df[x_column],
        df["baseline_priority_coverage_percent"],
        marker="o",
        label="Baseline",
        color="#d95f02",
    )
    if "kmeans_priority_coverage_percent" in df.columns:
        axes[1].plot(
            df[x_column],
            df["kmeans_priority_coverage_percent"],
            marker="o",
            label="K-Means",
            color="#4c78a8",
        )
    axes[1].plot(
        df[x_column],
        df["pso_priority_coverage_percent"],
        marker="o",
        label="PSO",
        color="#1b9e77",
    )
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("Priority coverage (%)")
    axes[1].set_title("Priority-zone coverage")
    axes[1].grid(True, linestyle=":", alpha=0.35)
    axes[1].legend()

    fig.suptitle(title)
    st.pyplot(fig)
    plt.close(fig)


def show_scenario_chart(df: pd.DataFrame) -> None:
    """Compare baseline and PSO gains across user-distribution scenarios."""
    fig, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    x_values = range(len(df))
    width = 0.35
    axis.bar(
        [value - width / 2 for value in x_values],
        df["connected_users_gain_pp"],
        width=width,
        label="Connected-user gain",
        color="#4c78a8",
    )
    axis.bar(
        [value + width / 2 for value in x_values],
        df["priority_coverage_gain_pp"],
        width=width,
        label="Priority-coverage gain",
        color="#9467bd",
    )
    axis.set_xticks(list(x_values))
    axis.set_xticklabels(df["scenario"], rotation=10)
    axis.set_ylabel("PSO gain over baseline (percentage points)")
    axis.set_title("PSO improvement by user-distribution scenario")
    axis.grid(axis="y", linestyle=":", alpha=0.35)
    axis.legend()
    st.pyplot(fig)
    plt.close(fig)


def show_reliability_chart(df: pd.DataFrame) -> None:
    """Show repeated-run distributions using box plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    method_labels = ["Random/static", "K-Means", "PSO"]
    method_colors = ["#d95f02", "#4c78a8", "#1b9e77"]

    connected_data = [
        df["baseline_connected_users_percent"],
        df["kmeans_connected_users_percent"],
        df["pso_connected_users_percent"],
    ]
    priority_data = [
        df["baseline_priority_coverage_percent"],
        df["kmeans_priority_coverage_percent"],
        df["pso_priority_coverage_percent"],
    ]
    _colored_boxplot(
        axes[0],
        connected_data,
        method_labels,
        method_colors,
        "Connected-user reliability",
        "Connected users (%)",
    )
    _colored_boxplot(
        axes[1],
        priority_data,
        method_labels,
        method_colors,
        "Priority-coverage reliability",
        "Priority coverage (%)",
    )
    fig.suptitle("Repeated runs: method reliability across random seeds")
    st.pyplot(fig)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    gain_data = [
        df["kmeans_connected_users_gain_pp"] + df["kmeans_priority_coverage_gain_pp"],
        df["connected_users_gain_pp"] + df["priority_coverage_gain_pp"],
    ]
    _colored_boxplot(
        axis,
        gain_data,
        ["K-Means gain", "PSO gain"],
        ["#4c78a8", "#1b9e77"],
        "Total improvement over random/static baseline",
        "Connected-user + priority gain (percentage points)",
    )
    st.pyplot(fig)
    plt.close(fig)

    summary = pd.DataFrame(
        [
            {
                "Method": "Random/static",
                "Mean connected users (%)": df["baseline_connected_users_percent"].mean(),
                "Mean priority coverage (%)": df["baseline_priority_coverage_percent"].mean(),
            },
            {
                "Method": "K-Means",
                "Mean connected users (%)": df["kmeans_connected_users_percent"].mean(),
                "Mean priority coverage (%)": df["kmeans_priority_coverage_percent"].mean(),
            },
            {
                "Method": "PSO",
                "Mean connected users (%)": df["pso_connected_users_percent"].mean(),
                "Mean priority coverage (%)": df["pso_priority_coverage_percent"].mean(),
            },
        ]
    )
    st.caption("Box plots show spread across random seeds; the table gives mean values.")
    st.dataframe(summary.round(2), use_container_width=True, hide_index=True)


def _colored_boxplot(
    axis: plt.Axes,
    data: list[pd.Series],
    labels: list[str],
    colors: list[str],
    title: str,
    ylabel: str,
) -> None:
    """Draw a readable box plot with method colours."""
    box = axis.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", linestyle=":", alpha=0.35)


def show_dynamic_motion_chart(df: pd.DataFrame) -> None:
    """Show how placement performance changes as users move over time."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    for column, label, color in [
        ("baseline_connected_users_percent", "Random/static", "#d95f02"),
        ("kmeans_connected_users_percent", "K-Means", "#4c78a8"),
        ("pso_connected_users_percent", "PSO", "#1b9e77"),
    ]:
        axes[0].plot(df["time_step"], df[column], marker="o", label=label, color=color)
    axes[0].set_title("Connected users over time")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Connected users (%)")
    axes[0].grid(True, linestyle=":", alpha=0.35)
    axes[0].legend()

    for column, label, color in [
        ("baseline_priority_coverage_percent", "Random/static", "#d95f02"),
        ("kmeans_priority_coverage_percent", "K-Means", "#4c78a8"),
        ("pso_priority_coverage_percent", "PSO", "#1b9e77"),
    ]:
        axes[1].plot(df["time_step"], df[column], marker="o", label=label, color=color)
    axes[1].set_title("Priority coverage over time")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Priority coverage (%)")
    axes[1].grid(True, linestyle=":", alpha=0.35)
    axes[1].legend()

    axes[2].bar(df["time_step"], df["pso_step_movement_cost_m"], color="#1b9e77")
    axes[2].set_title("PSO movement per time step")
    axes[2].set_xlabel("Time step")
    axes[2].set_ylabel("Movement cost (m)")
    axes[2].grid(axis="y", linestyle=":", alpha=0.35)

    fig.suptitle("Dynamic scenario: moving users and UAV re-optimization")
    st.pyplot(fig)
    plt.close(fig)
    show_dynamic_uav_trails(df)


def show_dynamic_uav_trails(df: pd.DataFrame) -> None:
    """Draw PSO UAV trails across dynamic time steps."""
    drone_ids = sorted(
        {
            int(column.split("_")[1])
            for column in df.columns
            if column.startswith("drone_") and column.endswith("_x")
        }
    )
    if not drone_ids:
        return

    fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(drone_ids)))

    for drone_id, color in zip(drone_ids, colors):
        x_values = df[f"drone_{drone_id}_x"]
        y_values = df[f"drone_{drone_id}_y"]
        axis.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2,
            color=color,
            label=f"UAV {drone_id}",
        )
        axis.annotate(
            "",
            xy=(x_values.iloc[-1], y_values.iloc[-1]),
            xytext=(x_values.iloc[-2], y_values.iloc[-2]) if len(x_values) > 1 else (x_values.iloc[0], y_values.iloc[0]),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.8},
        )

    axis.set_title("PSO UAV movement trails over time")
    axis.set_xlabel("x position (m)")
    axis.set_ylabel("y position (m)")
    axis.grid(True, linestyle=":", alpha=0.35)
    axis.legend(loc="upper right", fontsize=8)
    st.pyplot(fig)
    plt.close(fig)


def show_weight_sensitivity_chart(df: pd.DataFrame) -> None:
    """Show objective-weight sensitivity as trade-offs, not just bars."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    profiles = df["weight_profile"].tolist()
    movement = df["pso_movement_cost_m"]
    size_values = 80 + 260 * (movement - movement.min()) / max(movement.max() - movement.min(), 1e-9)

    scatter = axes[0, 0].scatter(
        df["pso_connected_users_percent"],
        df["pso_priority_coverage_percent"],
        s=size_values,
        c=df["priority_main_report_score"],
        cmap="viridis",
        edgecolors="black",
        alpha=0.88,
    )
    for _, row in df.iterrows():
        axes[0, 0].annotate(
            row["weight_profile"],
            (row["pso_connected_users_percent"], row["pso_priority_coverage_percent"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axes[0, 0].set_title("Trade-off: user connectivity vs priority coverage")
    axes[0, 0].set_xlabel("Connected users (%)")
    axes[0, 0].set_ylabel("Priority coverage (%)")
    axes[0, 0].grid(True, linestyle=":", alpha=0.35)
    fig.colorbar(scatter, ax=axes[0, 0], label="Common priority-aware report score")

    axes[0, 1].plot(
        profiles,
        df["priority_main_report_score"],
        marker="o",
        linewidth=2.4,
        color="#1b9e77",
    )
    axes[0, 1].set_title("Comparable score using the main priority-aware formula")
    axes[0, 1].set_ylabel("report score")
    axes[0, 1].tick_params(axis="x", rotation=15)
    axes[0, 1].grid(axis="y", linestyle=":", alpha=0.35)

    bottom = np.zeros(len(df))
    for column, label, color in [
        ("CU_weight", "CU", OBJECTIVE_TERM_COLORS["CU"]),
        ("PZ_weight", "PZ", OBJECTIVE_TERM_COLORS["PZ"]),
        ("AC_weight", "AC", OBJECTIVE_TERM_COLORS["AC"]),
        ("RL_weight", "RL", OBJECTIVE_TERM_COLORS["RL"]),
        ("MC_weight", "MC", OBJECTIVE_TERM_COLORS["MC"]),
    ]:
        axes[1, 0].bar(profiles, df[column], bottom=bottom, label=label, color=color)
        bottom += df[column].to_numpy()
    axes[1, 0].set_title("Weight profiles being tested")
    axes[1, 0].set_ylabel("weight")
    axes[1, 0].tick_params(axis="x", rotation=15)
    axes[1, 0].legend(ncol=5, fontsize=8)
    axes[1, 0].grid(axis="y", linestyle=":", alpha=0.35)

    axes[1, 1].barh(profiles, df["pso_movement_cost_m"], color="#777777")
    axes[1, 1].set_title("Movement cost trade-off")
    axes[1, 1].set_xlabel("movement cost (m)")
    axes[1, 1].grid(axis="x", linestyle=":", alpha=0.35)

    fig.suptitle("Objective weight sensitivity: what changes when the formula changes?")
    st.pyplot(fig)
    plt.close(fig)

    best_row = df.loc[df["priority_main_report_score"].idxmax()]
    st.success(
        f"Using the common priority-aware report score, the strongest profile in this run is "
        f"**{best_row['weight_profile']}** with score **{best_row['priority_main_report_score']:.3f}**."
    )

    show_weight_profile_radar(df)


def show_weight_profile_radar(df: pd.DataFrame) -> None:
    """Display normalized trade-offs for each objective-weight profile."""
    metrics = [
        "pso_connected_users_percent",
        "pso_priority_coverage_percent",
        "pso_area_coverage_percent",
        "relay_link_ratio",
        "movement_efficiency_score",
    ]
    labels = [
        "Connected users",
        "Priority coverage",
        "Area coverage",
        "Relay links",
        "Low movement",
    ]

    values = df[metrics].copy()
    for column in metrics:
        column_min = values[column].min()
        column_max = values[column].max()
        if column_max > column_min:
            values[column] = (values[column] - column_min) / (column_max - column_min)
        else:
            values[column] = 1.0

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, axis = plt.subplots(figsize=(7, 6), subplot_kw={"polar": True})
    for row_index, row in values.iterrows():
        profile_values = row.tolist()
        profile_values += profile_values[:1]
        axis.plot(angles, profile_values, linewidth=2, label=df.loc[row_index, "weight_profile"])
        axis.fill(angles, profile_values, alpha=0.08)

    axis.set_xticks(angles[:-1])
    axis.set_xticklabels(labels)
    axis.set_yticklabels([])
    axis.set_title("Normalized trade-off diagram")
    axis.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10))
    st.pyplot(fig)
    plt.close(fig)


def run_dynamic_motion_analysis(
    config: SimulationConfig,
    swarm_size: int,
    iterations: int,
    time_steps: int = 4,
) -> pd.DataFrame:
    """Move users over time and re-optimize UAV placement at each step."""
    simulation = DisasterNetworkSimulation(config)
    rng = np.random.default_rng(config.random_seed + 10)

    users = simulation.generate_users()
    baseline_drones = simulation.generate_drones()
    previous_pso_drones = baseline_drones.copy()
    priority_zones = simulation.generate_priority_zones()
    rows = []

    # This is a dashboard demo, so the dynamic run is capped to stay responsive.
    # Longer motion experiments should be run offline and reported separately.
    dynamic_swarm_size = min(swarm_size, 12)
    dynamic_iterations = min(iterations, 15)

    for time_step in range(time_steps):
        kmeans_drones = kmeans_drone_placement(simulation, users, baseline_drones)

        optimizer = ParticleSwarmOptimizer(
            simulation=simulation,
            users=users,
            initial_drones=previous_pso_drones,
            priority_zones=priority_zones,
            swarm_size=dynamic_swarm_size,
            iterations=dynamic_iterations,
        )
        pso_drones, best_score, _ = optimizer.optimize()

        baseline_graph = simulation.build_network_graph(users, baseline_drones)
        kmeans_graph = simulation.build_network_graph(users, kmeans_drones)
        pso_graph = simulation.build_network_graph(users, pso_drones)

        baseline_metrics = evaluate_solution(
            config, users, baseline_drones, baseline_drones, baseline_graph, priority_zones
        )
        kmeans_metrics = evaluate_solution(
            config, users, kmeans_drones, baseline_drones, kmeans_graph, priority_zones
        )
        pso_metrics = evaluate_solution(
            config, users, pso_drones, previous_pso_drones, pso_graph, priority_zones
        )
        add_improvement_metrics(baseline_metrics, kmeans_metrics)
        add_improvement_metrics(baseline_metrics, pso_metrics)

        row = {
            "time_step": time_step,
            "baseline_connected_users_percent": baseline_metrics[
                "connected_users_percent"
            ],
            "kmeans_connected_users_percent": kmeans_metrics[
                "connected_users_percent"
            ],
            "pso_connected_users_percent": pso_metrics[
                "connected_users_percent"
            ],
            "baseline_priority_coverage_percent": baseline_metrics[
                "priority_coverage_percent"
            ],
            "kmeans_priority_coverage_percent": kmeans_metrics[
                "priority_coverage_percent"
            ],
            "pso_priority_coverage_percent": pso_metrics[
                "priority_coverage_percent"
            ],
            "pso_step_movement_cost_m": pso_metrics["movement_cost"],
            "best_pso_score": best_score,
        }
        for drone_index, drone in enumerate(pso_drones):
            row[f"drone_{drone_index}_x"] = drone[0]
            row[f"drone_{drone_index}_y"] = drone[1]
        rows.append(row)

        previous_pso_drones = pso_drones
        # Simple random-walk motion: users shift slightly between time steps.
        user_motion = rng.normal(0.0, 35.0, size=users.shape)
        users = simulation.clip_positions(users + user_motion)

    return pd.DataFrame(rows)


def run_weight_sensitivity_analysis(
    config: SimulationConfig,
    swarm_size: int,
    iterations: int,
) -> pd.DataFrame:
    """Run PSO with several objective-weight profiles on the same scenario."""
    rows = []
    # Sensitivity analysis checks whether conclusions depend too much on one
    # chosen objective formula.
    sensitivity_swarm_size = min(swarm_size, 30)
    sensitivity_iterations = min(iterations, 70)

    for profile_name, weights in WEIGHT_PROFILES.items():
        result = run_experiment(
            config,
            swarm_size=sensitivity_swarm_size,
            iterations=sensitivity_iterations,
            objective_weights=weights,
        )
        relay_ratio = calculate_relay_ratio(result.optimized_graph, config.num_drones)
        movement_ratio = result.optimized_metrics["movement_cost"] / (
            config.num_drones * np.hypot(config.area_width, config.area_height)
        )
        connected_ratio = result.optimized_metrics["connected_users_percent"] / 100.0
        priority_ratio = result.optimized_metrics["priority_coverage_percent"] / 100.0
        area_ratio = result.optimized_metrics["coverage_percent"] / 100.0
        priority_main_report_score = (
            OBJECTIVE_WEIGHTS["connected_users"] * connected_ratio
            + OBJECTIVE_WEIGHTS["priority_coverage"] * priority_ratio
            + OBJECTIVE_WEIGHTS["area_coverage"] * area_ratio
            + OBJECTIVE_WEIGHTS["relay_links"] * relay_ratio
            - OBJECTIVE_WEIGHTS["movement_cost"] * movement_ratio
        )
        rows.append(
            {
                "weight_profile": profile_name,
                "CU_weight": weights["connected_users"],
                "PZ_weight": weights["priority_coverage"],
                "AC_weight": weights["area_coverage"],
                "RL_weight": weights["relay_links"],
                "MC_weight": weights["movement_cost"],
                "pso_connected_users_percent": result.optimized_metrics[
                    "connected_users_percent"
                ],
                "pso_priority_coverage_percent": result.optimized_metrics[
                    "priority_coverage_percent"
                ],
                "pso_area_coverage_percent": result.optimized_metrics[
                    "coverage_percent"
                ],
                "relay_link_ratio": relay_ratio,
                "pso_movement_cost_m": result.optimized_metrics["movement_cost"],
                "movement_efficiency_score": max(0.0, 1.0 - movement_ratio),
                "best_pso_score": result.best_score,
                "priority_main_report_score": priority_main_report_score,
            }
        )

    return pd.DataFrame(rows)


def calculate_relay_ratio(graph, num_drones: int) -> float:
    """Calculate the ratio of active UAV-to-UAV relay links."""
    drone_nodes = [f"drone_{index}" for index in range(num_drones)]
    relay_edges = graph.subgraph(drone_nodes).number_of_edges()
    max_relay_edges = max(1, num_drones * (num_drones - 1) / 2)
    return float(relay_edges / max_relay_edges)


def run_uav_sweep(
    config: SimulationConfig, swarm_size: int, iterations: int
) -> pd.DataFrame:
    """Run simulations for different UAV counts."""
    rows = []
    for index, num_drones in enumerate([2, 3, 4, 5, 6, 8, 10]):
        sweep_config = replace_config(
            config,
            num_drones=num_drones,
            random_seed=config.random_seed + index,
        )
        result = run_experiment(sweep_config, swarm_size=swarm_size, iterations=iterations)
        rows.append(flatten_result(index + 1, result, config.scenario_type))
    return pd.DataFrame(rows)


def run_area_sweep(
    config: SimulationConfig, swarm_size: int, iterations: int
) -> pd.DataFrame:
    """Run simulations for different square disaster-area sizes."""
    rows = []
    for index, area_size in enumerate([500, 750, 1000, 1250, 1500, 2000]):
        area_config = replace_config(
            config,
            area_width=float(area_size),
            area_height=float(area_size),
            random_seed=config.random_seed + index,
        )
        result = run_experiment(area_config, swarm_size=swarm_size, iterations=iterations)
        row = flatten_result(index + 1, result, config.scenario_type)
        row["area_size_m"] = area_size
        rows.append(row)
    return pd.DataFrame(rows)


def run_range_sweep(
    config: SimulationConfig, swarm_size: int, iterations: int
) -> pd.DataFrame:
    """Run simulations for different user-to-UAV communication ranges."""
    rows = []
    for index, user_range in enumerate([100, 140, 180, 220, 260, 300]):
        range_config = replace_config(
            config,
            user_range=float(user_range),
            random_seed=config.random_seed + index,
        )
        result = run_experiment(range_config, swarm_size=swarm_size, iterations=iterations)
        row = flatten_result(index + 1, result, config.scenario_type)
        row["user_range_m"] = user_range
        rows.append(row)
    return pd.DataFrame(rows)


def run_iteration_sweep(config: SimulationConfig, swarm_size: int) -> pd.DataFrame:
    """Run simulations for different PSO iteration counts."""
    rows = []
    for index, iteration_count in enumerate([10, 30, 60, 90, 150]):
        iteration_config = replace_config(config, random_seed=config.random_seed + index)
        result = run_experiment(
            iteration_config,
            swarm_size=swarm_size,
            iterations=iteration_count,
        )
        row = flatten_result(index + 1, result, config.scenario_type)
        row["iterations"] = iteration_count
        rows.append(row)
    return pd.DataFrame(rows)


def run_scenario_comparison(
    config: SimulationConfig, swarm_size: int, iterations: int
) -> pd.DataFrame:
    """Run simulations for each user-distribution scenario."""
    rows = []
    scenarios = ["Random users", "Clustered users", "Hotspot priority zones"]
    for index, scenario_type in enumerate(scenarios):
        scenario_config = replace_config(
            config,
            scenario_type=scenario_type,
            random_seed=config.random_seed + index,
        )
        result = run_experiment(scenario_config, swarm_size=swarm_size, iterations=iterations)
        rows.append(flatten_result(index + 1, result, scenario_type))
    return pd.DataFrame(rows)


def replace_config(config: SimulationConfig, **updates: object) -> SimulationConfig:
    """Create a copy of the simulation config with selected values changed."""
    values = {
        "area_width": config.area_width,
        "area_height": config.area_height,
        "num_users": config.num_users,
        "num_drones": config.num_drones,
        "uav_altitude": config.uav_altitude,
        "user_range": config.user_range,
        "drone_range": config.drone_range,
        "random_seed": config.random_seed,
        "scenario_type": config.scenario_type,
        "priority_source": config.priority_source,
        "obstacles_enabled": config.obstacles_enabled,
    }
    values.update(updates)
    return SimulationConfig(**values)


if __name__ == "__main__":
    main()
