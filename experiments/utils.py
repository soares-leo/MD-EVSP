import os
import sys
import time
import datetime

# Import project modules
from initializer.inputs import *
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

def build_connection_network(timetables_path, used_depots, experiment_name):
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

    metrics = ["graph_build_time", "number_of_nodes", "number_of_edges"]
    log_ids = [f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}{i}" for i in range(len(metrics))]
    values = [build_time, graph.number_of_nodes(), graph.number_of_edges()]
    log_df = pd.DataFrame({
        "log_ids": log_ids,
        "experiment_name": [experiment_name] * len(metrics),
        "step": ["graph_build"] * len(metrics),
        "metric": metrics,
        "value": values,
    })
    
    return graph, log_df
