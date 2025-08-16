"""
Column Generation Algorithm - Refactored Version
This module implements a column generation algorithm for vehicle scheduling optimization.
"""

import os
import sys
import time
import json
import datetime
import numpy as np
import pandas as pd
import pprint
from collections import deque, namedtuple
from datetime import timedelta

# Import project modules
from initializer.inputs import *
from initializer.generator import Generator
from initializer.utils import *
from initializer.conn_network_builder import GraphBuilder
from column_generator.utils import add_reduced_cost_info
from column_generator.rmp_solver import run_rmp
from column_generator.spfa_v9 import run_spfa

# ============================================================================
# CONFIGURATION SECTION - All main parameters in CAPS
# ============================================================================

# Paths and Files
CPLEX_PATH = "C:/Program Files/IBM/ILOG/CPLEX_Studio2211/cplex/bin/x64_win64/cplex.exe"
DEFAULT_INSTANCE_PATH = None
ALTERNATIVE_INSTANCE_PATH = "initializer/files/instance_2025-08-16_00-10-31_l4_290_l59_100_l60_120.csv"

# Algorithm Parameters
DEADHEAD_SPEED = 20  # km/h for deadhead calculations
RANDOM_SEED = 1
MAX_DRIVING_TIME = 120  # D parameter in minutes
MAX_TOTAL_TIME = 16 * 60  # T_d parameter in minutes

# Experiment Sets Configuration
NUM_EXPERIMENT_SETS = 30  # Number of experiment sets to run
K_VALUES_PER_SET = [50, 50, 50]  # K values to test in each set
TMAX_LIST = [10.0, 15.0, 30.0]

# Generate experiment set IDs and notes automatically
EXP_SET_IDS = [f"experiment_set_{i+1:03d}" for i in range(NUM_EXPERIMENT_SETS)]
EXP_NOTES = [f"Experiment set {i+1} with K values {K_VALUES_PER_SET}" for i in range(NUM_EXPERIMENT_SETS)]

# Experiment Parameters for each experiment within a set
EXPERIMENTS_PER_SET = len(K_VALUES_PER_SET)  # 3 experiments per set
FILTER_GRAPH = False  # Same for all experiments
Z_MIN = 50  # Convergence threshold
MAX_ITER = 9  # Maximum iterations

# ============================================================================
# UTILITY CLASSES
# ============================================================================

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


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

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
        tmax=tmax
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


def create_experiment_name(exp_number, z_min, k, max_iter, filter_graph, instance_name):
    """
    Create a standardized experiment name.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filter_str = "filtered" if filter_graph else "unfiltered"
    
    # Extract key instance info and sanitize it for use in a directory path
    # THIS LINE IS THE FIX: It replaces forbidden ":" characters with "-"
    instance_info = instance_name.replace("instance_", "").replace(".csv", "").replace(":", "-")
    
    return f"exp{exp_number:03d}_{timestamp}_z{z_min}_k{k}_i{max_iter}_{filter_str}_{instance_info}"


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


def generate_columns(S, graph, depots, dh_df, dh_times_df, z_min, k, max_iter, 
                    exp_number, filter_graph, timetables_path, exp_set_id, 
                    instance_name):
    """
    Main column generation algorithm.
    
    Returns:
        tuple: (final_columns, min_objective, z_values, column_counts, results_df)
    """
    
    experiment_name = create_experiment_name(
        exp_number, z_min, k, max_iter, filter_graph, instance_name
    )
    
    exp_dir = f"experiments/{exp_set_id}/{experiment_name}"
    os.makedirs(exp_dir, exist_ok=True)
    
    log_filename = f"{exp_dir}/experiment.log"
    logger = Logger(log_filename)
    original_stdout = sys.stdout
    sys.stdout = logger
    
    print("="*80)
    print(f"EXPERIMENT: {experiment_name}")
    print("="*80)
    print(f"Parameters:")
    print(f"  - Z_min (convergence threshold): {z_min}")
    print(f"  - K (columns per iteration): {k}")
    print(f"  - Max iterations: {max_iter}")
    print(f"  - Graph filtering: {filter_graph}")
    print(f"  - Instance: {instance_name}")
    print("="*80)
    
    # Initialize tracking variables
    cnt = 0
    it = 1
    iterations = []
    z_values = []
    differences = []
    column_counts = []
    iteration_times = []
    rmp_times = []
    graph_times = []
    spfa_times = []
    label_times = []
    filter_times = []
    
    while cnt <= max_iter:
        iteration_start = time.time()
        
        print(f"\n{'='*80}")
        print(f"ITERATION {it:03d}")
        print(f"{'='*80}")
        
        # Solve Restricted Master Problem
        print("\n► Solving RMP...")
        rmp_start = time.time()
        
        status, model, current_cost, duals = run_rmp(
            depots, S, 
            timetables_csv=timetables_path,
            cplex_path=CPLEX_PATH
        )
        
        rmp_time = time.time() - rmp_start
        rmp_times.append(rmp_time)
        print(f"  RMP solved in {rmp_time:.2f}s")
        print(f"  Objective value: {current_cost:.2f}")
        
        # Save RMP model
        with open(f"{exp_dir}/RMP_iteration_{it:03d}.txt", "w") as f:
            f.write(model.__str__())
        
        # Check convergence
        last_z = z_values[-1] if z_values else current_cost
        z_values.append(current_cost)
        diff = abs(current_cost - last_z)
        differences.append(diff)
        
        if diff < z_min:
            cnt += 1
            print(f"  Convergence counter: {cnt}/{max_iter+1}")
        else:
            cnt = 0
        
        print(f"  Improvement: {diff:.2f}")
        
        # Calculate reduced costs
        print("\n► Computing reduced costs...")
        graph_start = time.time()
        red_cost_graph = add_reduced_cost_info(graph, duals, dh_times_df)
        graph_time = time.time() - graph_start
        graph_times.append(graph_time)
        print(f"  Reduced cost graph computed in {graph_time:.2f}s")
        
        # Run SPFA
        print("\n► Running SPFA...")
        trip_keys = list(map(lambda x: int(x.split("_")[-1]), list(S.keys())))
        last_route_number = max(trip_keys)
        source_nodes = [n for n, d in graph.nodes(data=True) if d.get("type")=="K"]
        
        all_labels = {}
        spfa_start = time.time()
        
        for i, t in enumerate(source_nodes):
            print(f"  Source {t} ({i+1}/{len(source_nodes)})...", end="")
            node_start = time.time()
            
            all_labels[t] = run_spfa(
                red_cost_graph, t, 
                D=MAX_DRIVING_TIME, 
                T_d=MAX_TOTAL_TIME, 
                dh_df=dh_df, 
                duals=duals, 
                filter_graph=filter_graph
            )
            
            node_time = time.time() - node_start
            print(f" {node_time:.2f}s")
        
        spfa_time = time.time() - spfa_start
        spfa_times.append(spfa_time)
        print(f"  Total SPFA time: {spfa_time:.2f}s")
        
        # Label correction and filtering
        print("\n► Processing labels...")
        label_start = time.time()
        
        flat = {}
        for depot, routes in all_labels.items():
            for rkey, info in routes.items():
                if flat.get(rkey) is None:
                    flat[rkey] = []
                for data in info:
                    flat[rkey].append(data)
        
        filtered_routes = filter_routes_by_dominance(flat)
        
        label_time = time.time() - label_start
        label_times.append(label_time)
        print(f"  Label processing time: {label_time:.2f}s")
        
        # Select top K columns
        print("\n► Selecting top columns...")
        filter_start = time.time()
        
        unique_routes = list({tuple(route['Path']): route for route in filtered_routes}.values())
        unique_routes.sort(key=lambda x: x["ReducedCost"])
        topk = unique_routes[:k]
        
        # Add new columns
        new_columns_added = 0
        print(f"\n  Top {k} columns with negative reduced cost:")
        
        for i, info in enumerate(topk, start=last_route_number+1):
            if info["ReducedCost"] >= 0:
                continue
                
            new_key = f"Route_{i}"
            S[new_key] = {
                "Path": info["Path"],
                "Cost": info["Cost"],
                "Data": info["Data"]
            }
            new_columns_added += 1
            
            if new_columns_added <= 5:  # Show first 5 for readability
                path_str = " → ".join(info['Path'][:5])
                if len(info['Path']) > 5:
                    path_str += f" ... ({len(info['Path'])} nodes)"
                print(f"    {new_key}: RC={info['ReducedCost']:.2f}, Path={path_str}")
        
        print(f"  Added {new_columns_added} new columns")
        
        filter_time = time.time() - filter_start
        filter_times.append(filter_time)
        
        column_counts.append(len(S))
        iteration_time = time.time() - iteration_start
        iteration_times.append(iteration_time)
        iterations.append(f"{it:03d}")
        
        print(f"\n  Iteration {it} completed in {iteration_time:.2f}s")
        print(f"  Total columns: {len(S)}")
        
        it += 1
    
    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETED")
    print(f"{'='*80}")
    
    # Create results dataframe
    results = pd.DataFrame({
        'Iteration': iterations,
        'Experiment': [experiment_name] * len(iterations),
        'Time': [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * len(iterations),
        'Objective': [round(z, 2) for z in z_values],
        'Columns': column_counts,
        'Difference': [round(d, 2) for d in differences],
        'RMP_time': [round(t, 4) for t in rmp_times],
        'Graph_time': [round(t, 4) for t in graph_times],
        'SPFA_time': [round(t, 4) for t in spfa_times],
        'Label_time': [round(t, 4) for t in label_times],
        'Filter_time': [round(t, 4) for t in filter_times],
        'Iteration_time': [round(t, 4) for t in iteration_times]
    })
    
    print("\nResults Summary:")
    print(results[["Iteration", "Objective", "Columns", "Difference", "Iteration_time"]])
    print(f"\nFinal objective value: {min(z_values):.2f}")
    print(f"Total solving time: {sum(iteration_times):.2f} seconds")
    print(f"Total columns generated: {len(S)}")
    
    # Save results
    results.to_csv(f"{exp_dir}/iteration_results.csv", index=False)
    
    sys.stdout = original_stdout
    logger.close()
    
    return S, min(z_values), z_values, column_counts, results, experiment_name


def save_columns_data(columns, exp_set_id, experiment_name):
    """Save the columns dictionary to a Python file."""
    
    output_filename = f"experiments/{exp_set_id}/{experiment_name}/columns_data.py"
    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("import pandas as pd\n")
        f.write("import numpy as np\n\n")
        f.write("columns = ")
        pprint.pprint(columns, stream=f, indent=4)
    
    print(f"✓ Columns data saved to: {output_filename}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main(tmax_values_per_set=None,
         base_instance_path=None,
         num_experiment_sets=None,
         k_values_per_set=None,
         filter_graph=None,
         z_min=None,
         max_iter=None):
    """
    Main function to run the column generation algorithm.
    
    This version creates multiple experiment sets, each with its own instance.
    Within each set, different initial solutions are generated using different tmax values,
    but all experiments use the same underlying timetables/graph.
    
    Args:
        tmax_values_per_set: List of tmax values for initial solution generation
        base_instance_path: Base path for instances (if None, generates new ones)
        num_experiment_sets: Number of experiment sets to run
        k_values_per_set: List of K values to test in each set
        filter_graph: Graph filtering option (same for all)
        z_min: Convergence threshold (same for all)
        max_iter: Maximum iterations (same for all)
    """
    
    # Use default values if not provided
    tmax_values_per_set = tmax_values_per_set or TMAX_VALUES_PER_SET
    num_experiment_sets = num_experiment_sets or NUM_EXPERIMENT_SETS
    k_values_per_set = k_values_per_set or K_VALUES_PER_SET
    filter_graph = filter_graph if filter_graph is not None else FILTER_GRAPH
    z_min = z_min or Z_MIN
    max_iter = max_iter or MAX_ITER
    
    # Validate that tmax and k lists have the same length
    if len(tmax_values_per_set) != len(k_values_per_set):
        raise ValueError(f"tmax_values_per_set and k_values_per_set must have the same length. "
                        f"Got {len(tmax_values_per_set)} and {len(k_values_per_set)}")
    
    print("\n" + "="*80)
    print("COLUMN GENERATION ALGORITHM - BATCH EXECUTION")
    print("="*80)
    print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Number of experiment sets: {num_experiment_sets}")
    print(f"Experiments per set: {len(k_values_per_set)}")
    print(f"K values per set: {k_values_per_set}")
    print(f"Tmax values per set: {tmax_values_per_set}")
    print(f"Total experiments: {num_experiment_sets * len(k_values_per_set)}")
    print("="*80 + "\n")
    
    # Initialize base data structures (shared across all experiments)
    dh_df, dh_times_df = initialize_data()
    
    # Track overall timing and results
    total_start_time = time.time()
    all_sets_results = []
    
    # Main loop over experiment sets
    for set_idx in range(num_experiment_sets):
        set_number = set_idx + 1
        exp_set_id = f"experiment_set_{set_number:03d}"
        exp_note = f"Experiment set {set_number} with K values {k_values_per_set} and Tmax values {tmax_values_per_set}"
        
        print("\n" + "="*80)
        print(f"EXPERIMENT SET {set_number}/{num_experiment_sets}: {exp_set_id}")
        print("="*80)
        print(f"Notes: {exp_note}")
        
        # Create experiment set directory
        os.makedirs(f"experiments/{exp_set_id}", exist_ok=True)
        
        # Generate a NEW instance (timetables) for this experiment set
        print(f"\n► Generating new instance for set {set_number}...")
        
        # Use the first tmax value for initial instance generation
        gen, initial_solution_first, used_depots, instance_name, gen_time = generate_instance(
            tmax_values_per_set[0], None
        )
        
        # Save the instance path for reference
        instance_path_for_set = gen.timetables_path
        
        # Filter depots
        filtered_depots = {key: value for key, value in depots.items() if key in used_depots}
        
        # Build connection network for this instance (shared across all experiments in the set)
        graph, graph_build_time = build_connection_network(gen.timetables_path, used_depots)
        
        # Generate initial solutions for other tmax values using the SAME timetables
        initial_solutions = [initial_solution_first]
        initial_solution_gen_times = [0]  # First one already generated
        
        for tmax_idx in range(1, len(tmax_values_per_set)):
            print(f"\n► Generating initial solution {tmax_idx + 1}/{len(tmax_values_per_set)} for set {set_number}")
            init_sol, init_gen_time = generate_initial_solution_with_tmax(
                gen, tmax_values_per_set[tmax_idx]
            )
            initial_solutions.append(init_sol)
            initial_solution_gen_times.append(init_gen_time)
        
        # Save experiment set info
        with open(f"experiments/{exp_set_id}/experiment_info.txt", "w") as f:
            f.write(f"Experiment Set: {exp_set_id}\n")
            f.write(f"Set Number: {set_number}/{num_experiment_sets}\n")
            f.write(f"Instance: {instance_name}\n")
            f.write(f"Instance Path: {instance_path_for_set}\n")
            f.write(f"Instance Generation Time: {gen_time:.2f}s\n")
            f.write(f"Graph Build Time: {graph_build_time:.2f}s\n")
            f.write(f"K Values: {k_values_per_set}\n")
            f.write(f"Tmax Values: {tmax_values_per_set}\n")
            f.write(f"Filter Graph: {filter_graph}\n")
            f.write(f"Z_min: {z_min}\n")
            f.write(f"Max Iterations: {max_iter}\n")
            f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"\nNotes:\n{exp_note}\n")
        
        set_results = pd.DataFrame()
        set_summary = {
            'set_number': set_number,
            'exp_set_id': exp_set_id,
            'instance_name': instance_name,
            'instance_path': instance_path_for_set,
            'experiments': []
        }
        
        # Run experiments within this set
        for exp_idx, (k_value, tmax_value) in enumerate(zip(k_values_per_set, tmax_values_per_set)):
            exp_number = exp_idx + 1
            
            print(f"\n► Running Experiment {exp_number}/{len(k_values_per_set)} in Set {set_number}")
            print(f"  Parameters: k={k_value}, tmax={tmax_value}, z_min={z_min}, max_iter={max_iter}, filter={filter_graph}")
            print(f"  Initial solution size: {len(initial_solutions[exp_idx])} routes")
            
            columns, optimal, z_vals, col_counts, results, exp_name = generate_columns(
                S=initial_solutions[exp_idx].copy(),  # Use the corresponding initial solution
                graph=graph,  # Same graph for all experiments in this set
                depots=filtered_depots,
                dh_df=dh_df,
                dh_times_df=dh_times_df,
                z_min=z_min,
                k=k_value,
                max_iter=max_iter,
                exp_number=exp_number,
                filter_graph=filter_graph,
                timetables_path=gen.timetables_path,  # Same timetables for all experiments in set
                exp_set_id=exp_set_id,
                instance_name=instance_name
            )
            
            # Save columns data
            save_columns_data(columns, exp_set_id, exp_name)
            
            # Track experiment results
            total_exp_time = results['Iteration_time'].sum() + initial_solution_gen_times[exp_idx]
            exp_summary = {
                'exp_number': exp_number,
                'k_value': k_value,
                'tmax_value': tmax_value,
                'initial_solution_size': len(initial_solutions[exp_idx]),
                'optimal_value': optimal,
                'final_columns': len(columns),
                'iterations': len(z_vals),
                'initial_solution_gen_time': initial_solution_gen_times[exp_idx],
                'column_gen_time': results['Iteration_time'].sum(),
                'total_time': total_exp_time
            }
            set_summary['experiments'].append(exp_summary)
            
            # Add set and experiment identifiers to results
            results['Set_Number'] = set_number
            results['K_Value'] = k_value
            results['Tmax_Value'] = tmax_value
            
            # Merge results
            set_results = pd.concat([set_results, results], ignore_index=True)
        
        # Save set results
        set_results.to_csv(f"experiments/{exp_set_id}/all_results.csv", index=False)
        
        
        # Create set summary
        summary_df = pd.DataFrame({
            'Experiment': [f"Exp_{e['exp_number']}" for e in set_summary['experiments']],
            'K_Value': [e['k_value'] for e in set_summary['experiments']],
            'Optimal_Value': [e['optimal_value'] for e in set_summary['experiments']],
            'Final_Columns': [e['final_columns'] for e in set_summary['experiments']],
            'Iterations': [e['iterations'] for e in set_summary['experiments']],
            'Total_Time': [e['total_time'] for e in set_summary['experiments']],
            'Initial_Solution_Gen_Time': [e['initial_solution_gen_time'] for e in set_summary['experiments']],
            'Column_Gen_Time': [e['column_gen_time'] for e in set_summary['experiments']]
        })
        
        summary_df.to_csv(f"experiments/{exp_set_id}/summary.csv", index=False)
        
        print(f"\n✓ Experiment set {set_number} completed")
        print(f"  Instance: {instance_name}")
        print(f"  Results summary:")
        print(summary_df.to_string(index=False))
        
        all_sets_results.append(set_summary)
    
    # Create overall summary
    print("\n" + "="*80)
    print("CREATING OVERALL SUMMARY")
    print("="*80)
    
    overall_summary = []
    for set_data in all_sets_results:
        for exp in set_data['experiments']:
            overall_summary.append({
                'Set_Number': set_data['set_number'],
                'Set_ID': set_data['exp_set_id'],
                'Instance': set_data['instance_name'],
                'K_Value': exp['k_value'],
                'Optimal_Value': exp['optimal_value'],
                'Final_Columns': exp['final_columns'],
                'Total_Time': exp['total_time']
            })
    
    overall_df = pd.DataFrame(overall_summary)
    overall_df.to_csv("experiments/overall_summary.csv", index=False)
    
    # Create aggregated statistics
    stats_df = overall_df.groupby('K_Value').agg({
        'Optimal_Value': ['mean', 'std', 'min', 'max'],
        'Final_Columns': ['mean', 'std', 'min', 'max'],
        'Total_Time': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    stats_df.to_csv("experiments/aggregated_statistics.csv")
    
    print("\nAggregated Statistics by K Value:")
    print(stats_df)
    
    total_time = time.time() - total_start_time
    
    print("\n" + "="*80)
    print("ALL EXPERIMENT SETS COMPLETED")
    print("="*80)
    print(f"Total experiment sets: {num_experiment_sets}")
    print(f"Total experiments: {num_experiment_sets * len(k_values_per_set)}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Average time per set: {total_time/num_experiment_sets:.2f} seconds")
    print(f"End time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Return the last experiment's results for compatibility
    return columns, optimal


# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run 30 experiment sets, each with 3 experiments using different K values
    # Each set generates its own instance that is shared among its 3 experiments
    
    final_columns, final_objective = main(
        tmax_values_per_set=TMAX_LIST,
        base_instance_path=None,  # None means generate new instances
        num_experiment_sets=NUM_EXPERIMENT_SETS,  # 30 sets
        k_values_per_set=K_VALUES_PER_SET,  # [10, 30, 50] for each set
        filter_graph=FILTER_GRAPH,  # True
        z_min=Z_MIN,  # 50
        max_iter=MAX_ITER  # 9
    )
    
    print("Program finished successfully!")
    print(f"Check 'experiments/' directory for all results and summaries.")