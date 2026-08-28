# AeroRelief-Sim Architecture

This document describes the current architecture of **AeroRelief-Sim**, a
priority-aware UAV placement simulator for disaster communication.

## Scope

The project is a focused UAV placement simulator for disaster communication research. It evaluates where UAV relay nodes should be positioned to improve temporary communication coverage when normal infrastructure is unavailable.

The simulator includes:

- ground users/rescue nodes;
- UAV relay nodes at configurable altitude;
- synthetic or real OpenStreetMap priority zones;
- optional rectangular 3D obstacles;
- graph-based communication links;
- random/static, K-Means, and PSO placement methods;
- dynamic user-motion analysis;
- dashboard-based experiments and CSV export.

It is not intended to replace packet-level simulators such as NS-3. Packet delivery ratio, latency, and throughput can be added later through NS-3 export or simplified traffic models.

## Objects

### Users

Users represent rescue workers, affected civilians, or mobile devices requiring communication support. They are represented as ground-level `(x, y)` points. Scenario options control user distribution:

- random users;
- clustered users;
- hotspot users near priority zones.

### UAVs

UAVs are temporary aerial relay nodes. Their placement is represented by ground coordinates `(x, y)` plus a fixed altitude from the simulation configuration.

The current simulator models placement and communication coverage, not detailed flight physics.

### Priority Zones

Priority zones represent locations that should receive stronger communication support. They have:

- name;
- centre point;
- radius;
- priority weight.

The simulator supports:

- synthetic emergency zones;
- real OpenStreetMap Stirling priority locations from `data/stirling_priority_zones_osm.csv`.

### Obstacles

Obstacles are optional rectangular 3D blocks with height. They represent simplified buildings or blocked structures. A ground user cannot connect to a UAV if the straight line from user to UAV passes through an obstacle below the UAV altitude.

Obstacles are not priority zones. Priority zones represent communication demand; obstacles represent physical blockage. For example, a hospital may be a priority zone, while a nearby damaged building may be an obstacle.

## Communication Model

User-to-UAV links use 3D distance:

```text
3D distance = sqrt(horizontal distance^2 + UAV altitude^2)
```

A user-to-UAV link exists when:

- the 3D distance is within user-to-UAV communication range;
- line of sight is not blocked by an enabled obstacle.

UAV-to-UAV links exist when UAV horizontal distance is within relay range.

The full network is represented as a NetworkX graph.

## Placement Methods

### Random/static

UAVs are randomly placed inside the disaster area. This is the no-optimization baseline.

### K-Means

Users are clustered using a simple NumPy K-Means implementation. UAVs are placed near user-cluster centres. This gives a stronger non-PSO baseline because it uses user distribution but does not optimize priority zones, relay links, or movement cost directly.

### PSO

Particle Swarm Optimization searches for UAV positions that maximize a weighted objective function. Each particle represents one complete UAV deployment.

## Objective Function

```text
Score = 0.40(CU) + 0.35(PZ) + 0.10(AC) + 0.10(RL) - 0.05(MC)
```

Where:

- `CU`: connected-user ratio;
- `PZ`: priority-zone coverage ratio;
- `AC`: area coverage ratio;
- `RL`: UAV relay-link ratio;
- `MC`: normalized movement cost.

The absolute weights add up to 1. Movement cost is subtracted because large repositioning distance is undesirable. The main formula is priority-aware because priority-zone support is central to the research topic.

## Weight Sensitivity

The dashboard can rerun PSO using different objective-weight profiles:

- priority-aware main;
- user-focused;
- coverage-focused;
- energy-aware.

The output is shown as both a table and graphs, including connected users, priority coverage, movement cost, objective score, and a normalized trade-off diagram. This helps test whether the placement conclusions depend too strongly on one selected weighting scheme.

## Dynamic Scenario

The dynamic analysis simulates changing demand over time:

1. Generate users and UAVs.
2. Evaluate random/static, K-Means, and PSO placement.
3. Move users slightly using a random-walk motion model.
4. Re-optimize UAV placement at the next time step.
5. Plot connected users, priority coverage, and PSO movement cost over time.

This is a dynamic user-motion scenario with UAV re-optimization, not full UAV flight dynamics.

## Metrics

- connected users percentage;
- area coverage percentage;
- weighted priority-zone coverage;
- number of network fragments;
- average path length in hops;
- UAV movement cost in metres;
- PSO objective score.
- objective-weight sensitivity results.

## Result Graphs

```text
results_comparison.png       2D random/K-Means/PSO deployment map
results_metrics.png          Core metric comparison
results_efficiency.png       Movement cost and efficiency
results_pso_convergence.png  PSO objective score over iterations
results_3d_deployment.png    Fixed-altitude 3D deployment view
```

## Code Structure

```text
main.py            Command-line entry point.
app.py             Streamlit dashboard.
experiment.py      Full experiment flow and result object.
simulation.py      Users, UAVs, priority zones, obstacles, and graph links.
optimization.py    PSO and K-Means placement logic.
metrics.py         Evaluation metrics.
visualization.py   2D/3D plots and metric charts.
```
