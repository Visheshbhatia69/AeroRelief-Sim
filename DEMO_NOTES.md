# Demo Notes: AeroRelief-Sim

## One-Sentence Explanation

AeroRelief-Sim evaluates how UAV relay nodes can be placed after a disaster to improve temporary communication coverage, comparing random/static, K-Means, and PSO-based placement.

## What The Demo Shows

1. A disaster area is created.
2. Ground users/rescue nodes are generated.
3. Priority zones represent important emergency locations.
4. UAVs are placed using three methods:
   - random/static baseline;
   - K-Means user-cluster placement;
   - PSO objective-based placement.
5. UAVs operate at a fixed altitude.
6. User-to-UAV links use 3D distance.
7. Optional obstacles can block line of sight.
8. The system builds a graph-based communication network.
9. The dashboard compares connected users, priority coverage, network cohesion, path length, and movement cost.
10. The major-findings tab summarizes the main technical interpretation.
11. The objective contribution chart explains why the PSO result is better.
12. The before/after demand heatmap shows where communication need is concentrated and how much remains after PSO.

## Objective Function

```text
Score = 0.40(CU) + 0.35(PZ) + 0.10(AC) + 0.10(RL) - 0.05(MC)
```

The absolute weights add up to 1. Movement cost is a penalty. Priority-zone coverage is weighted strongly because the project focus is priority-aware UAV placement.

## Weight Sensitivity Explanation

The dashboard includes an objective-weight sensitivity analysis. It reruns PSO using different weight profiles:

- priority-aware main;
- user-focused;
- coverage-focused;
- energy-aware.

This evaluates whether conclusions are dependent on one selected objective-function weighting:

> Do the results depend too much on one chosen set of objective-function weights?

The results are presented as tables, bar charts, and a normalized trade-off diagram.

## Dynamic Scenario Explanation

Dynamic means that user positions change over time. In the dashboard, users move slightly at each time step and UAV positions are re-optimized. This tests whether UAV placement can adapt to changing disaster communication demand.

Technical description:

> This is a dynamic user-motion scenario with UAV re-optimization, not a full flight-dynamics model.

## 3D And Obstacle Explanation

The simulator uses fixed-altitude UAVs. Users and priority zones remain on the ground. A user-to-UAV link is calculated using 3D distance:

```text
3D distance = sqrt(horizontal distance^2 + altitude^2)
```

When obstacles are enabled, rectangular building-like blocks can block line of sight between users and UAVs.

Technical description:

> This is a fixed-altitude 3D communication model with simplified obstacle-aware line-of-sight blocking.

## Current Technical Coverage

The work now goes beyond a static 2D placement demo. It includes:

- algorithm comparison: random/static, K-Means, and PSO;
- normalized objective-function weights;
- real OpenStreetMap priority-zone data;
- fixed-altitude 3D communication;
- obstacle-aware line-of-sight blocking;
- dynamic user-motion analysis;
- objective-weight sensitivity analysis;
- objective contribution diagram;
- before/after demand heatmap for spatial interpretation;
- sensitivity analysis and CSV export.

The simulator remains focused on UAV placement decisions rather than packet-level networking. NS-3 integration is a suitable next step for packet delivery, latency, and throughput validation.

## Heatmap Explanation

The heatmap is a diagnostic layer calculated from users, weighted priority zones, and UAV coverage. It asks:

> Where was communication demand concentrated before optimization, and how much of that need remains after PSO placement?

The score is based on:

- user concentration;
- weighted priority-zone concentration;
- UAV communication coverage after PSO;
- communication range, altitude, and obstacle blocking.

Technical description:

> This is a before/after demand-relief heatmap. It shows how PSO reduces the most communication-needy parts of the disaster area.

## Project Summary

> The project provides a focused UAV placement simulator for disaster communication planning. It demonstrates how AI-based placement can improve connected users and priority-zone support compared with random/static and K-Means baselines, while also considering altitude, obstacles, dynamic user demand, relay links, and movement cost.
