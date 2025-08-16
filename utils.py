import os
import sys
import time
import datetime
import pprint

# Import project modules
from initializer.inputs import *
from initializer.generator import Generator
from initializer.utils import *
from initializer.conn_network_builder import GraphBuilder
from config import *

class Logger:
    """Custom logger that writes to both console and file."""
    
    def __init__(self, filename="output.log"):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

def save_columns_data(columns, exp_set_id, experiment_name):
    """Save the columns dictionary to a Python file."""
    
    output_filename = f"experiments/{exp_set_id}/{experiment_name}/columns_data.py"
    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("import pandas as pd\n")
        f.write("import numpy as np\n\n")
        f.write("columns = ")
        pprint.pprint(columns, stream=f, indent=4)
    
    print(f"✓ Columns data saved to: {output_filename}")

def initialize_data():

    """Initialize all base data structures needed for the algorithm."""
    
    print("="*80)
    print("INITIALIZING BASE DATA STRUCTURES")
    print("="*80)
    
    cp_locations_summary = summarize_cp_locations(lines_info)
    cp_distances = calculate_cp_distances(cp_locations_summary, lines_info)
    transformed_cp_depot_distances = transform_cp_depot_distances(cp_depot_distances)
    dh_dict = merge_distances_dicts(transformed_cp_depot_distances, cp_distances)
    dh_df = make_deadhead_df(dh_dict)
    dh_times_df = make_deadhead_times_df(DEADHEAD_SPEED, dh_df)
    
    print("✓ Data structures initialized successfully\n")
    
    return dh_df, dh_times_df


def generate_instance(tmax, instance_path=None):
    """
    Generate or load an instance for the optimization problem.
    
    Args:
        tmax: Tmax value for initial solution generation
        instance_path: Path to existing instance. If None, generates a new one.
        
    Returns:
        tuple: (generator, initial_solution, used_depots, instance_name, generation_time)
    """
    
    print("="*80)
    print("INSTANCE GENERATION/LOADING")
    print("="*80)
    
    start_time = time.time()
    
    # CORRECTED LOGIC: Now it will correctly show "Generating new instance"
    # when instance_path is None.
    print(f"Instance path: {instance_path if instance_path else 'Generating new instance'}")
    
    # By passing 'instance_path' (which is None), we tell the Generator to create a new file.
    gen = Generator(
        lines_info, 
        cp_depot_distances, 
        depots, 
        timetables_path_to_use=instance_path, # FIX IS HERE
        seed=RANDOM_SEED,
        tmax=tmax,
        trips_configs=TRIPS_CONFIGS
    )
    
    print("Generating initial solution...")
    initial_solution, used_depots, instance_name = gen.generate_initial_set()
    
    generation_time = time.time() - start_time
    
    print(f"✓ Instance ready: {instance_name}")
    print(f"✓ Generation time: {generation_time:.2f} seconds")
    print(f"✓ Used depots: {list(used_depots)}")
    print(f"✓ Initial solution size: {len(initial_solution)} routes\n")
    
    return gen, initial_solution, used_depots, instance_name, generation_time


def build_connection_network(timetables_path, used_depots):
    """
    Build the connection network graph.
    
    Returns:
        tuple: (graph, build_time)
    """
    
    print("="*80)
    print("BUILDING CONNECTION NETWORK")
    print("="*80)
    
    start_time = time.time()
    
    graph_builder = GraphBuilder(timetables_path, used_depots)
    graph = graph_builder.build_graph()
    
    build_time = time.time() - start_time
    
    print(f"✓ Graph built successfully")
    print(f"✓ Nodes: {graph.number_of_nodes()}")
    print(f"✓ Edges: {graph.number_of_edges()}")
    print(f"✓ Build time: {build_time:.2f} seconds\n")
    
    return graph, build_time

def generate_initial_solution_with_tmax(gen, tmax):
    """
    Generate initial solution with a specific tmax value using an existing generator.
    
    Args:
        gen: Generator instance with loaded timetables
        tmax: Maximum time parameter for initial solution generation
        
    Returns:
        tuple: (initial_solution, generation_time)
    """
    
    print(f"  Generating initial solution with tmax={tmax}...")
    start_time = time.time()
    
    # Update the generator's tmax parameter
    gen.tmax = tmax
    
    # Generate initial solution with the new tmax
    initial_solution, used_depots, instance_name = gen.generate_initial_set()
    
    generation_time = time.time() - start_time
    print(f"  ✓ Initial solution generated in {generation_time:.2f}s")
    print(f"  ✓ Solution size: {len(initial_solution)} routes")
    
    return initial_solution, generation_time


def create_experiment_name(exp_number, z_min, k, max_iter, filter_graph, instance_name, tmax):
    """
    Create a standardized experiment name.
    """
    filter_str = "filt" if filter_graph else "unfi"
    
    # Extract key instance info and sanitize it for use in a directory path
    # THIS LINE IS THE FIX: It replaces forbidden ":" characters with "-"
    instance_info = instance_name.replace("instance_", "").replace(".csv", "").replace(":", "-")
    
    return f"exp{exp_number:03d}_z{z_min}_k{k}_i{max_iter}_tmax{tmax}_{filter_str}_{instance_info}"


def filter_routes_by_dominance(flat):
    """
    Filter routes based on dominance criteria.
    
    A route is discarded if another route has lower/equal cost 
    AND higher/equal distance.
    """
    
    new_flat = []
    
    for k, routes_in_group in flat.items():
        if not routes_in_group:
            continue

        routes_to_keep = []
        
        for route_b in routes_in_group:
            is_discarded = False
            
            for route_a in routes_in_group:
                if route_a is route_b:
                    continue

                if (route_a["ReducedCost"] <= route_b["ReducedCost"] and 
                    route_a["Data"]["total_travel_dist"] >= route_b["Data"]["total_travel_dist"]):
                    is_discarded = True
                    break

            if not is_discarded:
                routes_to_keep.append(route_b)

        new_flat.extend(routes_to_keep)
    
    return new_flat