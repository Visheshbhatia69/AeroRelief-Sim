# AeroRelief-Sim: A Priority-Aware UAV Placement Simulator for Disaster Communication

This project implements **AeroRelief-Sim**, a UAV placement simulator for
temporary disaster communication support.

The simulator models a disaster area where normal communication infrastructure is unavailable. Ground users represent rescue workers, mobile users, or affected civilians. UAVs act as temporary aerial relay nodes. The system compares random/static UAV placement, K-Means user-cluster placement, and Particle Swarm Optimization (PSO).

## Project Aim

The aim is to evaluate whether priority-aware AI-based UAV placement can improve emergency communication support by increasing:

- connected users;
- priority-zone coverage;
- overall area coverage;
- UAV relay connectivity;
- movement efficiency.

The simulator includes fixed-altitude 3D communication, optional obstacle-aware line-of-sight blocking, dynamic user-motion experiments, real OpenStreetMap priority-zone data, and a Streamlit dashboard for controlled experiments.

## Main Features

- 2D disaster map with fixed-altitude UAVs.
- Altitude-aware 3D user-to-UAV communication distance.
- Optional 3D rectangular obstacles that can block line of sight.
- Random/static UAV placement baseline.
- K-Means user-cluster placement baseline.
- PSO objective-based UAV placement.
- Synthetic and real OpenStreetMap priority-zone data.
- Dynamic user-motion analysis over time.
- Sensitivity analysis for UAV count, range, area size, PSO iterations, and scenarios.
- Objective-weight sensitivity analysis for testing whether results depend on chosen formula weights.
- Major-findings dashboard tab for result interpretation.
- Before/after communication-demand heatmap showing how PSO reduces needy areas.
- Dashboard visualizations and CSV export.

## Project Structure

```text
main.py            Runs the command-line simulation and saves figures.
app.py             Streamlit dashboard for interactive simulation control.
experiment.py      Runs one full experiment and stores all outputs.
simulation.py      Creates users, UAVs, priority zones, obstacles, and graph links.
optimization.py    Implements PSO and K-Means UAV placement.
metrics.py         Calculates connectivity, coverage, graph, and movement metrics.
visualization.py   Creates 2D, 3D, metric, and convergence plots.
requirements.txt   Python dependencies.
ARCHITECTURE.md    Technical design explanation.
DEMO_NOTES.md      Short demonstration and explanation notes.
```

## How The Simulator Works

1. A square disaster area is created.
2. Ground users are generated using a selected scenario: random, clustered, or hotspot-based.
3. Priority zones are created using either synthetic zones or OpenStreetMap Stirling data.
4. UAVs are placed using random/static, K-Means, and PSO methods.
5. UAVs fly at a configurable fixed altitude.
6. User-to-UAV links use 3D distance:

```text
3D distance = sqrt(horizontal distance^2 + UAV altitude^2)
```

7. Optional obstacles block line of sight between ground users and UAVs.
8. UAV-to-UAV links are created when UAVs are within relay range.
9. The network is represented as a NetworkX graph.
10. Metrics and visualizations compare the placement methods.

## Objective Function

PSO uses a normalized weighted objective:

```text
Score = 0.40(CU) + 0.35(PZ) + 0.10(AC) + 0.10(RL) - 0.05(MC)
```

Where:

- `CU`: connected-user ratio.
- `PZ`: priority-zone coverage ratio.
- `AC`: area coverage ratio.
- `RL`: UAV relay-link ratio.
- `MC`: normalized UAV movement cost.

The absolute weights add up to 1. Movement cost is subtracted because excessive repositioning is undesirable. Priority-zone coverage is weighted strongly because the project focus is priority-aware UAV placement.

The dashboard includes objective-weight sensitivity analysis using multiple profiles:

- priority-aware main;
- user-focused;
- coverage-focused;
- energy-aware.

This tests whether conclusions are stable or overly dependent on one chosen set of weights.

## Placement Methods

**Random/static placement**  
Initial baseline. UAVs are placed randomly without optimization.

**K-Means placement**  
UAVs are placed near user cluster centres. This is a simple clustering baseline.

**PSO placement**  
Particle Swarm Optimization searches for UAV positions that maximize the weighted objective function.

The PSO search is seeded with both the random/static deployment and the K-Means deployment. This makes the comparison fairer because PSO is allowed to improve from a strong clustering baseline rather than relying only on random initial particles.

## Metrics

**Connected users (%)**  
Percentage of users connected to at least one UAV-supported network component.

**Area coverage (%)**  
Estimated percentage of the map covered by UAV ground communication footprints.

**Priority-zone coverage (weighted %)**  
Weighted percentage of hospitals, shelters, rescue centres, or real OSM priority locations covered by UAV communication.

**Network fragments**  
Number of connected graph components. Lower usually means a less fragmented communication network.

**Network cohesion score (%)**  
Graph-friendly version of network fragments where higher is better. It converts connected-component counts into a relative score for comparison charts.

**Average path length (hops)**  
Average shortest path length inside the largest connected component.

**Movement cost (m)**  
Total UAV repositioning distance from the initial random/static positions.

## Presentation Outputs

The dashboard supports simulation, result inspection, and evaluation-output generation. The main outputs are:

- deployment maps showing random/static, K-Means, and PSO UAV placement;
- a slope chart showing how connected users and priority-zone coverage change across methods;
- a network-cohesion chart where higher values mean the network is less fragmented;
- PSO convergence showing whether the AI search improves over iterations;
- an objective contribution chart showing which weighted terms explain the PSO result;
- a before/after communication-demand heatmap showing needy areas before deployment and remaining uncovered demand after PSO;
- sensitivity-analysis line graphs showing how results change when UAV count, range, area size, PSO iterations, scenario type, or objective weights are changed.

The heatmap is not treated as a separate prediction model. It is an explanation layer built from the same simulator assumptions. The first panel shows demand concentration from user density and weighted priority zones. The second panel shows the remaining demand after PSO UAV coverage is applied using communication range, altitude, and optional obstacle blocking.

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the command-line simulation:

```bash
python main.py
```

Run the dashboard:

```bash
streamlit run app.py
```

Alternative local launch script:

```text
run_dashboard.bat
```

## Output Files

The command-line simulation saves:

```text
results_comparison.png
results_metrics.png
results_efficiency.png
results_pso_convergence.png
results_3d_deployment.png
```

The dashboard saves temporary images inside:

```text
dashboard_outputs/
```

## Real Data

The simulator includes a real OpenStreetMap priority-zone sample:

```text
data/stirling_priority_zones_osm.csv
```

This provides real hospital, police, fire, health, and community locations for the Stirling area. User positions are still simulated so experiments can be controlled and repeated.

## Priority Zones vs Obstacles

Priority zones and obstacles are separate object types:

- priority zones are important locations that need communication support, such as hospitals, shelters, rescue centres, or real OSM emergency locations;
- obstacles are physical barriers, such as damaged buildings, collapsed structures, industrial obstructions, or dense urban blocks.

A hospital may need coverage, while a nearby damaged building may block line of sight. The simulator keeps these roles separate so service demand is not confused with physical blockage.

## Scope And Limitations

This is a focused UAV placement simulator, not a full packet-level network simulator. It is suitable for studying placement decisions, coverage, graph connectivity, priority-zone support, and movement cost.

Final project position:

- the project is a placement-optimisation and decision-support simulator;
- it is not claiming to model full packet delivery, throughput, or real UAV flight control;
- the strongest research evidence comes from comparing placement methods and showing sensitivity analysis;
- NS-3 or another network simulator can be used later for packet-level validation if needed.

Current limitations:

- no packet delivery ratio, throughput, or latency model yet;
- fixed UAV altitude rather than full flight dynamics;
- simplified rectangular obstacles;
- simplified random-walk user motion;
- no battery discharge or UAV speed limit yet.

## Future Work

- Export scenarios to NS-3 for packet-level validation.
- Add realistic mobility traces.
- Add UAV speed, battery, and hovering-energy models.
- Add line-of-sight/path-loss radio modelling.
- Compare with genetic algorithms or reinforcement learning.
- Add no-fly zones and more detailed map layers.
