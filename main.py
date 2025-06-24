"""
Traffic Simulation for Kathmandu, Nepal

This script simulates traffic flow in Kathmandu using OpenStreetMap data.
It includes traffic lights, vehicle movements, and visualization.

Features:
- Automatic caching of downloaded map data to avoid re-downloading
- Traffic light simulation at major intersections
- Vehicle arrival and movement simulation
- Real-time visualization

Usage:
  python main.py              - Run simulation
  python main.py --clear-cache - Clear cached graph data
  python main.py --cache-info  - Show cache information
  python main.py --help        - Show help

The first run will download and cache the Kathmandu road network data.
Subsequent runs will use the cached data for faster startup.
"""

import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
import heapq
import time
import os
import pickle
import sys

# --- CONFIGURATION ---
CITY = "Kathmandu, Nepal"
CACHE_DIR = "graph_cache"  # Directory to store cached graph data
CACHE_FILE = f"{CACHE_DIR}/kathmandu_graph.pkl"
SIM_DURATION = 3600  # seconds (1 hour)
VEHICLE_ARRIVAL_RATE_PER_MIN = 30  # vehicles per minute (Poisson)
ARRIVAL_LAMBDA = VEHICLE_ARRIVAL_RATE_PER_MIN / 60.0  # vehicles/sec
VEHICLE_SPEED_MPS = 10  # 36 km/h approx

# Traffic light cycle times (seconds)
GREEN_TIME = 30
YELLOW_TIME = 5
RED_TIME = 35  # Green + Yellow on other road

# Intersection degree threshold for traffic light assignment
TRAFFIC_LIGHT_THRESHOLD = 3  # intersections with >=3 edges get traffic lights

# --- Helper classes ---

class Vehicle:
    def __init__(self, vid, path, enter_time):
        self.id = vid
        self.path = path  # List of nodes to traverse
        self.current_index = 0  # current node index on path
        self.enter_time = enter_time
        self.position = 0  # meters along current edge
        self.speed = VEHICLE_SPEED_MPS
        self.status = "moving"  # moving, waiting, finished
        self.wait_start = None

    def current_edge(self):
        if self.current_index + 1 < len(self.path):
            return (self.path[self.current_index], self.path[self.current_index + 1])
        else:
            return None

    def advance(self, dt, G, traffic_lights):
        if self.status == "finished":
            return
        edge = self.current_edge()
        if edge is None:
            self.status = "finished"
            return

        # Handle multigraph - get the first edge between the two nodes
        edge_data = G.edges[edge[0], edge[1], 0] if G.is_multigraph() else G.edges[edge]
        edge_length = edge_data.get('length', 50)  # default length if missing

        # Check traffic light at next node if exists
        next_node = edge[1]
        tl = traffic_lights.get(next_node)
        if tl and self.position >= edge_length - 5:  # close to intersection
            if not tl.can_pass(edge[0], edge[1]):
                # Wait at intersection
                if self.status != "waiting":
                    self.status = "waiting"
                    self.wait_start = time.time()
                return  # Can't move forward yet
            else:
                if self.status == "waiting":
                    self.status = "moving"

        # Move forward
        self.position += self.speed * dt
        if self.position >= edge_length:
            # Move to next edge
            self.current_index += 1
            self.position = 0
            if self.current_index == len(self.path) - 1:
                self.status = "finished"


class TrafficLight:
    def __init__(self, node_id):
        self.node_id = node_id
        self.cycle_time = 0.0
        self.state = "green"  # green or red for simplicity
        self.last_switch = 0.0

    def update(self, current_time):
        # Simple fixed cycle: green 30s, red 35s (includes yellow for simplicity)
        cycle_pos = (current_time - self.last_switch) % (GREEN_TIME + RED_TIME)
        previous_state = self.state
        if cycle_pos < GREEN_TIME:
            self.state = "green"
        else:
            self.state = "red"
        # State changed?
        if previous_state != self.state:
            self.last_switch = current_time

    def can_pass(self, from_node, to_node):
        # If green, vehicles can pass
        return self.state == "green"


# --- Simulator ---

class TrafficSimulator:
    def __init__(self, G):
        self.G = G
        self.vehicles = []
        self.time = 0.0
        self.event_queue = []
        self.vehicle_id_counter = 1
        self.traffic_lights = {}
        self.setup_traffic_lights()
        self.arrival_lambda = ARRIVAL_LAMBDA
        self.completed_vehicles = []

    def setup_traffic_lights(self):
        for node, degree in self.G.degree():
            if degree >= TRAFFIC_LIGHT_THRESHOLD:
                self.traffic_lights[node] = TrafficLight(node)

    def schedule_vehicle_arrival(self, t):
        heapq.heappush(self.event_queue, (t, "arrival"))

    def run(self, sim_duration):
        self.schedule_vehicle_arrival(0)
        dt = 1  # 1 second time step for vehicle movement

        while self.time < sim_duration:
            # Process events
            while self.event_queue and self.event_queue[0][0] <= self.time:
                event_time, event_type = heapq.heappop(self.event_queue)
                if event_type == "arrival":
                    self.create_vehicle(event_time)
                    # Schedule next arrival
                    next_arrival = event_time + np.random.exponential(1/self.arrival_lambda)
                    self.schedule_vehicle_arrival(next_arrival)

            # Update traffic lights
            for tl in self.traffic_lights.values():
                tl.update(self.time)

            # Update vehicles
            for vehicle in list(self.vehicles):
                vehicle.advance(dt, self.G, self.traffic_lights)
                if vehicle.status == "finished":
                    self.completed_vehicles.append(vehicle)
                    self.vehicles.remove(vehicle)

            self.time += dt

    def create_vehicle(self, t):
        # Choose random origin and destination from graph nodes
        nodes = list(self.G.nodes())
        origin = random.choice(nodes)
        dest = random.choice(nodes)
        while dest == origin:
            dest = random.choice(nodes)
        try:
            path = nx.shortest_path(self.G, origin, dest, weight='length')
        except nx.NetworkXNoPath:
            return  # no path found, skip

        vehicle = Vehicle(self.vehicle_id_counter, path, t)
        self.vehicle_id_counter += 1
        self.vehicles.append(vehicle)

    def get_vehicle_positions(self):
        positions = []
        for v in self.vehicles:
            edge = v.current_edge()
            if edge is None:
                continue
            start_node_pos = (self.G.nodes[edge[0]]['x'], self.G.nodes[edge[0]]['y'])
            end_node_pos = (self.G.nodes[edge[1]]['x'], self.G.nodes[edge[1]]['y'])
            edge_length = self.G.edges[edge].get('length', 50)
            frac = v.position / edge_length
            x = start_node_pos[0] + frac * (end_node_pos[0] - start_node_pos[0])
            y = start_node_pos[1] + frac * (end_node_pos[1] - start_node_pos[1])
            positions.append((x, y))
        return positions


# --- Visualization ---

def plot_simulation(G, sim: TrafficSimulator):

    fig, ax = plt.subplots(figsize=(12,12))
    ox.plot_graph(G, ax=ax, show=False, close=False, node_size=10, edge_color='gray')

    scat = ax.scatter([], [], s=20, c='red')

    def update(frame):
        sim.time = frame
        # Update traffic lights
        for tl in sim.traffic_lights.values():
            tl.update(sim.time)

        # Advance simulation for 1 second
        dt = 1
        for vehicle in list(sim.vehicles):
            vehicle.advance(dt, sim.G, sim.traffic_lights)
            if vehicle.status == "finished":
                sim.completed_vehicles.append(vehicle)
                sim.vehicles.remove(vehicle)

        pos = sim.get_vehicle_positions()
        if pos:
            xs, ys = zip(*pos)
        else:
            xs, ys = [], []
        scat.set_offsets(np.c_[xs, ys])
        ax.set_title(f"Time: {sim.time:.0f}s, Vehicles: {len(sim.vehicles)}")
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, SIM_DURATION, 1), interval=50, blit=True)
    plt.show()


# --- Graph Loading with Caching ---

def load_or_download_graph(city, cache_file):
    """
    Load graph from cache if it exists, otherwise download and cache it.
    
    Args:
        city (str): City name for OSM query
        cache_file (str): Path to cache file
    
    Returns:
        networkx.MultiDiGraph: The road network graph
    """
    # Create cache directory if it doesn't exist
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    
    # Check if cached graph exists
    if os.path.exists(cache_file):
        print(f"Loading cached road network for {city}...")
        try:
            with open(cache_file, 'rb') as f:
                G = pickle.load(f)
            print("✓ Successfully loaded cached graph")
            return G
        except Exception as e:
            print(f"⚠ Error loading cached graph: {e}")
            print("Proceeding to download fresh data...")
    
    # Download and cache the graph
    print(f"Downloading and preparing road network for {city}...")
    try:
        G = ox.graph_from_place(city, network_type='drive')
        G = ox.project_graph(G)  # Project to UTM for meters accuracy
        
        # Save to cache        print("Saving graph to cache for future use...")
        with open(cache_file, 'wb') as f:
            pickle.dump(G, f)
        print("✓ Graph successfully cached")
        
    except Exception as e:
        print(f"✗ Error downloading graph: {e}")
        raise
    
    return G


def clear_graph_cache(cache_file):
    """
    Clear the cached graph file.
    
    Args:
        cache_file (str): Path to cache file
    """
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"✓ Cache cleared: {cache_file}")
    else:
        print("No cache file found to clear.")


def get_cache_info(cache_file):
    """
    Get information about the cached graph file.
    
    Args:
        cache_file (str): Path to cache file
    
    Returns:
        dict: Cache information
    """
    if os.path.exists(cache_file):
        stat = os.stat(cache_file)
        size_mb = stat.st_size / (1024 * 1024)
        mod_time = time.ctime(stat.st_mtime)
        return {
            'exists': True,
            'size_mb': round(size_mb, 2),
            'modified': mod_time,
            'path': cache_file
        }
    else:
        return {'exists': False}


# --- Main execution ---

def main():
    # Show cache information
    cache_info = get_cache_info(CACHE_FILE)
    if cache_info['exists']:
        print(f"📁 Cache found: {cache_info['size_mb']} MB, modified: {cache_info['modified']}")
    else:
        print("📁 No cache found, will download fresh data")
    
    # Load graph with caching
    G = load_or_download_graph(CITY, CACHE_FILE)
    
    print("Initializing traffic simulation...")
    sim = TrafficSimulator(G)
    
    print(f"Running simulation for {SIM_DURATION} seconds...")
    sim.run(SIM_DURATION)

    print(f"Simulation completed: {len(sim.completed_vehicles)} vehicles finished their trips.")

    print("Starting visualization...")
    plot_simulation(G, sim)

if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--clear-cache":
            clear_graph_cache(CACHE_FILE)
            sys.exit(0)
        elif sys.argv[1] == "--cache-info":
            cache_info = get_cache_info(CACHE_FILE)
            if cache_info['exists']:
                print(f"Cache file: {cache_info['path']}")
                print(f"Size: {cache_info['size_mb']} MB")
                print(f"Last modified: {cache_info['modified']}")
            else:
                print("No cache file found")
            sys.exit(0)
        elif sys.argv[1] == "--help":
            print("Traffic Simulation for Kathmandu")
            print("Usage:")
            print("  python main.py              - Run simulation")
            print("  python main.py --clear-cache - Clear cached graph data")
            print("  python main.py --cache-info  - Show cache information")
            print("  python main.py --help        - Show this help")
            sys.exit(0)
    
    main()
