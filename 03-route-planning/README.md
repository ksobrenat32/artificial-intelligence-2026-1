# Transit Route Planning System

A route planning system for Mexico City's public transportation network using Dijkstra's algorithm with transfer penalties. This system finds the fastest routes between stops across multiple transport methods including Metro, Metrobús, Trolebús, and more.

## Overview

This project implements an intelligent route planner that:

-   Processes GTFS (General Transit Feed Specification) data from Mexico City's transportation systems
-   Uses Dijkstra's algorithm with smart transfer penalties for optimal routing
-   Handles multiple transport methods and route types
-   Considers geographic proximity to merge nearby stops
-   Provides detailed route information including transfers and timing

## Features

### Multi-Modal Transportation Support

-   **Metro (Sistema de Transporte Colectivo Metro)**: 366 connections
-   **Metrobús**: 840 connections
-   **Red de Transporte de Pasajeros (RTP)**: 8,826 connections
-   **Trolebús**: 808 connections
-   **Pumabús**: 221 connections
-   **Cablebus**: 32 connections
-   **Tren Ligero**: 34 connections
-   **Ferrocarriles Suburbanos**: 12 connections
-   **Tren El Insurgente**: 8 connections

### Smart Routing Algorithm

-   **Dijkstra's Algorithm**: Finds optimal paths considering travel time
-   **Transfer Penalties**:
    -   8-minute penalty for changing between different transport methods
    -   8-minute penalty for changing routes within the same transport method
-   **Geographic Clustering**: Merges stops within 500m radius with same names
-   **Multi-Stop Support**: Handles duplicate stop names across different locations

### Route Information

-   Total travel time including penalties
-   Number of stops and transfers
-   Detailed step-by-step directions
-   Transport method and route information for each segment
-   Transfer indicators and penalty details

## Installation

### Setup

```bash
git clone https://github.com/ksobrenat32/artificial-intelligence-2026-1
cd 03-route-planning
```

## Usage

### Execution

```bash
python solve.py
```

## Algorithm Details

### Graph Construction

-   **Nodes**: Transit stops identified by `name@latitude,longitude`
-   **Edges**: Direct connections between stops with travel times
-   **Consolidation**: Stops with same names within 500m are merged

### Pathfinding

1. **State Representation**: `(stop_id, transport_method, route_short, route_long)`
2. **Transfer Detection**: Compare transport methods and routes between consecutive segments
3. **Penalty Application**: Add time penalties for transfers
4. **Optimization**: Dijkstra's algorithm finds minimum time path

### Transfer Logic

```python
# Transport method change: +8 minutes
if last_transport_method != new_transport_method:
    transfer_penalty = 8.0

# Route change within same transport method: +8 minutes
elif last_route != new_route:
    transfer_penalty = 8.0
```

## Technical Implementation

### Key Classes

#### `TransitGraph`

-   **Purpose**: Core graph data structure for transit network
-   **Methods**:
    -   `load_from_csv()`: Parse and load transit data
    -   `find_fastest_route()`: Main pathfinding interface
    -   `add_stop_info()`: Manage stop consolidation
    -   `_dijkstra_search()`: Core algorithm implementation

#### Core Functions

-   `display_route()`: Format and display route results
-   `get_user_input()`: Handle interactive user interface
-   `parse_time_to_minutes()`: Convert time formats

### Performance Characteristics

-   **Time Complexity**: O((V + E) log V) where V = stops, E = connections
-   **Space Complexity**: O(V + E) for graph storage
-   **Dataset Size**: ~11,000 connections across ~2,800 stops

## License

This project is part of an academic assignment for UNAM's Artificial Intelligence course.

## Acknowledgments

-   GTFS data provided by Mexico City transportation authorities
-   Built for UNAM Artificial Intelligence course 2026-1
-   Implements concepts from graph theory and shortest path algorithms
