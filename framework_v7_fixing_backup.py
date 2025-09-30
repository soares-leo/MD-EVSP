"""
Column Generation Algorithm - Refactored Version with Performance and Readability Improvements
This module implements an optimized column generation algorithm for vehicle scheduling.
"""

import os
import sys
import time
import datetime
import logging
import platform
import traceback
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from multiprocessing import Pool, cpu_count
import pickle

# Import project modules
from initializer.inputs import *
from initializer.utils import *
from column_generator.utils import add_reduced_cost_info
from column_generator.rmp_solver_v5_fixing import run_rmp
from column_generator.d_gemini_bom_spfa_v14 import run_spfa
from utils import *


# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

@dataclass
class ColumnGenConfig:
    """Configuration constants for column generation algorithm."""
    MAX_DISPLAY_COLUMNS: int = 5
    CONVERGENCE_WINDOW: int = 3
    INTEGRALITY_TOLERANCE: float = 1e-6
    MIN_REDUCED_COST: float = 0.0
    DEFAULT_NUM_PROCESSES: int = cpu_count()
    LOG_FORMAT: str = '%(asctime)s - %(levelname)s - %(message)s'


# ============================================================================
# PROGRESS TRACKING
# ============================================================================

class ProgressTracker:
    """Track and display progress information for column generation."""
    
    def __init__(self):
        self.iteration_data = defaultdict(list)
        self.start_time = time.time()
    
    def record(self, metric: str, value: float):
        """Record a metric value for the current iteration."""
        self.iteration_data[metric].append(value)
    
    def display_iteration(self, iteration: int, metrics: Dict[str, Any]):
        """Display formatted iteration summary."""
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration:03d}")
        print(f"{'='*80}")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    
    def get_elapsed_time(self) -> float:
        """Get total elapsed time since tracking started."""
        return time.time() - self.start_time


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(exp_dir: str, experiment_name: str) -> Tuple[logging.Logger, Any]:
    """
    Setup proper logging configuration.
    
    Returns:
        Tuple of (logger, original_stdout) for restoration
    """
    os.makedirs(exp_dir, exist_ok=True)
    log_file = f"{exp_dir}/column_generation.log"
    
    # Configure logger
    logger = Logger(log_file)
    original_stdout = sys.stdout
    sys.stdout = logger
    
    return logger, original_stdout


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_integrality(vars_values: Dict[str, float], 
                     tolerance: float = None) -> bool:
    """
    Check if all variable values are integral (0 or 1).
    
    Args:
        vars_values: Dictionary of variable names to values
        tolerance: Tolerance for checking integrality
    
    Returns:
        bool: True if all variables are integral, False otherwise
    """
    if tolerance is None:
        tolerance = ColumnGenConfig.INTEGRALITY_TOLERANCE
    
    for var_name, value in vars_values.items():
        if value is None or 'ARTIF' in var_name:
            continue
        # Check if value is close to 0 or 1
        if not (abs(value) < tolerance or abs(value - 1) < tolerance):
            return False
    return True


def fix_vars(vars_values: Dict[str, float]) -> List[str]:
    """
    Fix variables based on their values with improved logic.
    
    Args:
        vars_values: Dictionary of variable names to values
    
    Returns:
        List of variable names to fix
    """
    # Filter variables >= 0.9 first
    fixed_vars = [k for k, v in vars_values.items() if v >= 0.7]
    
    # If none found, use the maximum
    if not fixed_vars:
        max_var = max(vars_values, key=vars_values.get)
        fixed_vars = [max_var]
    
    print("\nFIXED VARIABLES:")
    for var in fixed_vars:
        print(f"  - {var}")
    print("END OF FIXED VARIABLES\n")
    
    return fixed_vars


def get_multiprocessing_context():
    """Get optimal multiprocessing context based on platform."""
    if platform.system() != 'Windows':
        return Pool  # Default fork on Unix/Linux
    else:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')  # Required on Windows
        return ctx.Pool


# ============================================================================
# SPFA PROCESSING
# ============================================================================

def run_spfa_wrapper(args: Tuple) -> Tuple:
    """
    Wrapper function for multiprocessing SPFA execution with improved error handling.
    
    Args:
        args: Tuple containing SPFA parameters
    
    Returns:
        Tuple of (source_node, labels, execution_time)
    """
    source_node = args[0]
    
    try:
        source_node, red_cost_graph, D, T_d, dh_df, duals, filter_graph = args
        
        start_time = time.time()
        labels = run_spfa(
            red_cost_graph, 
            source_node, 
            D=D, 
            T_d=T_d, 
            dh_df=dh_df, 
            duals=duals, 
            filter_graph=filter_graph
        )
        execution_time = time.time() - start_time
        
        return source_node, labels, execution_time
        
    except Exception as e:
        error_msg = f"SPFA failed for node {source_node}: {str(e)}\n{traceback.format_exc()}"
        print(f"  ⚠️ {error_msg}")
        return source_node, {}, 0.0


def process_spfa_results(all_labels: Dict, 
                        k: int, 
                        last_route_number: int, 
                        S: Dict) -> Tuple[int, float]:
    """
    Process SPFA results and add top-k columns to the solution set.
    
    Args:
        all_labels: Dictionary of SPFA labels
        k: Number of columns to add
        last_route_number: Last route number used
        S: Current solution set
    
    Returns:
        Tuple of (new_columns_added, total_reduced_cost)
    """
    # Flatten labels efficiently using defaultdict
    flat = defaultdict(list)
    for depot, routes in all_labels.items():
        for rkey, info in routes.items():
            flat[rkey].extend(info)
    
    # Filter by dominance
    filtered_routes = filter_routes_by_dominance(flat)
    
    # Get unique routes
    unique_routes = list({tuple(r['Path']): r for r in filtered_routes}.values())
    unique_routes.sort(key=lambda x: x["ReducedCost"])
    
    # Select and add top-k columns with negative reduced cost
    new_columns_added = 0
    total_reduced_cost = 0
    
    for i, info in enumerate(unique_routes[:k], start=last_route_number + 1):
        if info["ReducedCost"] >= ColumnGenConfig.MIN_REDUCED_COST:
            continue
        
        total_reduced_cost += info["ReducedCost"]
        S[f"Route_{i}"] = {
            "Path": info["Path"],
            "Cost": info["Cost"],
            "Data": info["Data"]
        }
        new_columns_added += 1
        
        # Display first few routes for readability
        if new_columns_added <= ColumnGenConfig.MAX_DISPLAY_COLUMNS:
            path_str = " → ".join(info['Path'][:5])
            if len(info['Path']) > 5:
                path_str += f" ... ({len(info['Path'])} nodes)"
    
    return new_columns_added, total_reduced_cost


# ============================================================================
# RESULTS PROCESSING
# ============================================================================

def create_results_dataframe(data_dict: Dict) -> pd.DataFrame:
    """
    Create results DataFrame with synchronized lengths.
    
    Args:
        data_dict: Dictionary of data lists
    
    Returns:
        pd.DataFrame with synchronized data
    """
    # Find minimum length across all lists
    list_lengths = [len(v) for v in data_dict.values() if isinstance(v, list)]
    if not list_lengths:
        return pd.DataFrame()
    
    min_len = min(list_lengths)
    
    # Truncate all lists to minimum length
    synchronized_data = {}
    for key, value in data_dict.items():
        if isinstance(value, list):
            synchronized_data[key] = value[:min_len]
        else:
            synchronized_data[key] = [value] * min_len
    
    return pd.DataFrame(synchronized_data)


def create_light_columns(S: Dict) -> Dict:
    """
    Create lightweight version of columns without 'Data' field.
    
    Args:
        S: Full solution set
    
    Returns:
        Dictionary with lightweight columns
    """
    S_light = {}
    for k, v in S.items():
        light_entry = {key: val for key, val in v.items() if key != "Data"}
        light_entry["op_time"] = (
            v["Data"]["total_dh_time"] + 
            v["Data"]["total_travel_time"] + 
            v["Data"]["total_wait_time"]
        )
        S_light[k] = light_entry
    
    return S_light


def extract_nonzero_columns(vars_values: Dict, 
                           var_to_route_mapping: Dict, 
                           S: Dict) -> Dict:
    """
    Extract columns with non-zero values from the solution.
    
    Args:
        vars_values: Variable values from solution
        var_to_route_mapping: Mapping from variables to routes
        S: Solution set
    
    Returns:
        Dictionary of non-zero columns
    """
    nonzero_cols = {}
    tolerance = ColumnGenConfig.INTEGRALITY_TOLERANCE
    
    for var_name, var_value in vars_values.items():
        # Check if the variable has a non-zero value and is not artificial
        if var_value > tolerance and 'ARTIF' not in var_name:
            # Find the original route key
            original_route_key = var_to_route_mapping.get(var_name)
            
            if original_route_key and original_route_key in S:
                # Copy the route data and add the solution value
                nonzero_cols[original_route_key] = S[original_route_key].copy()
                nonzero_cols[original_route_key]['var_value'] = var_value
    
    return nonzero_cols


# ============================================================================
# MAIN COLUMN GENERATION FUNCTION
# ============================================================================

def generate_columns(S, graph, depots, dh_df, dh_times_df, z_min, k, max_iter,
                    experiment_name, filter_graph, timetables_path, exp_set_id,
                    instance_name, use_multiprocessing=True):
    """
    Main column generation algorithm with improved structure and performance.
    
    Args:
        S: Initial solution set
        graph: Connection network graph
        depots: Dictionary of depots
        dh_df: Deadheading dataframe
        dh_times_df: Deadheading times dataframe
        z_min: Convergence threshold
        k: Number of columns to add per iteration
        max_iter: Maximum number of iterations
        experiment_name: Name of the experiment
        filter_graph: Graph filtering option
        timetables_path: Path to timetables
        exp_set_id: Experiment set ID
        instance_name: Instance name
        use_multiprocessing: Whether to use multiprocessing
    
    Returns:
        Comprehensive results tuple
    """
    
    # Initialize timing and configuration
    config = ColumnGenConfig()
    generate_columns_start_time = time.time()
    initial_num_of_columns = len(S)
    total_reduced_cost = 0
    
    # Setup logging
    exp_dir = f"experiments/{exp_set_id}/{experiment_name}"
    logger, original_stdout = setup_logging(exp_dir, experiment_name)
    
    # Initialize progress tracker
    tracker = ProgressTracker()
    
    # Print experiment header
    print("="*80)
    print(f"EXPERIMENT: {experiment_name}")
    print("="*80)
    print(f"Parameters:")
    print(f"  - Z_min (convergence threshold): {z_min}")
    print(f"  - K (columns per iteration): {k}")
    print(f"  - Max iterations: {max_iter}")
    print(f"  - Graph filtering: {filter_graph}")
    print(f"  - Instance: {instance_name}")
    print(f"  - Multiprocessing: {use_multiprocessing}")
    print("="*80)
    
    # Initialize tracking lists
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
    
    # Algorithm state
    cnt = 0
    it = 1
    is_optimal = False
    fixed_vars = []
    
    # Main column generation loop
    while True:
        iteration_start = time.time()
        
        # Display iteration header
        tracker.display_iteration(it, {"Status": "Starting"})
        
        # ========== SOLVE RMP ==========
        print("\n► Solving RMP...")
        rmp_start = time.time()
        
        status, model, current_cost, duals, vars_values, var_to_route_mapping = run_rmp(
            depots, S,
            timetables_csv=timetables_path,
            cplex_path=CPLEX_PATH,
            fixed_vars=fixed_vars
        )
        
        rmp_time = time.time() - rmp_start
        rmp_times.append(rmp_time)
        tracker.record("rmp_time", rmp_time)
        
        print(f"  RMP solved in {rmp_time:.2f}s")
        print(f"  Objective value: {current_cost:.2f}")
        
        # ========== CHECK CONVERGENCE ==========
        last_z = z_values[-1] if z_values else current_cost
        z_values.append(current_cost)
        diff = abs(current_cost - last_z)
        differences.append(diff)

        if diff < z_min:
            cnt += 1
            print(f"  Convergence counter: {cnt}/{max_iter}")
        else:
            cnt = 0
        print(f"  Convergence counter: {cnt}/{max_iter}")        
        print(f"  Improvement: {diff:.2f}")
        
        # ========== CHECK INTEGRALITY ==========
        if cnt == max_iter:
            if check_integrality(vars_values, config.INTEGRALITY_TOLERANCE):
                print("\n🥈🥈🥈 Solution is INTEGER! 🥈🥈🥈")
                
                # Add final values
                column_counts.append(len(S))
                iterations.append(f"{it:03d}")
                graph_times.append(0.0)
                spfa_times.append(0.0)
                label_times.append(0.0)
                filter_times.append(0.0)
                iteration_times.append(time.time() - iteration_start)
                break
            
            # Perform rounding
            fixed_vars = fix_vars(vars_values)
            cnt = 0
        
        # ========== COMPUTE REDUCED COSTS ==========
        print("\n► Computing reduced costs...")
        graph_start = time.time()
        red_cost_graph = add_reduced_cost_info(graph, duals, dh_times_df)
        graph_time = time.time() - graph_start
        graph_times.append(graph_time)
        tracker.record("graph_time", graph_time)
        print(f"  Reduced cost graph computed in {graph_time:.2f}s")
        
        # ========== RUN SPFA ==========
        trip_keys = list(map(lambda x: int(x.split("_")[-1]), list(S.keys())))
        last_route_number = max(trip_keys)
        source_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "K"]
        
        all_labels = {}
        spfa_start = time.time()
        
        if use_multiprocessing and len(source_nodes) > 1:
            print(f"\n► Running SPFA with multiprocessing ({len(source_nodes)} nodes)...")
            
            # Determine number of processes
            num_processes = min(config.DEFAULT_NUM_PROCESSES, len(source_nodes))
            
            try:
                # Prepare arguments
                mp_args = [
                    (t, red_cost_graph, MAX_DRIVING_TIME, MAX_TOTAL_TIME, 
                     dh_df, duals, filter_graph)
                    for t in source_nodes
                ]
                
                # Get optimal pool context
                pool_class = get_multiprocessing_context()
                
                # Execute in parallel
                with pool_class(processes=num_processes) as pool:
                    results = pool.map(run_spfa_wrapper, mp_args)
                
                # Collect results
                for source_node, labels, exec_time in results:
                    all_labels[source_node] = labels
                    
            except Exception as e:
                print(f"  ⚠️ Multiprocessing failed: {e}")
                print("  Falling back to sequential processing...")
                use_multiprocessing = False
        
        # Sequential fallback
        if not use_multiprocessing or len(source_nodes) == 1:
            print("\n► Running SPFA (sequential)...")
            
            for i, t in enumerate(source_nodes):
                all_labels[t] = run_spfa(
                    red_cost_graph, t,
                    D=MAX_DRIVING_TIME,
                    T_d=MAX_TOTAL_TIME,
                    dh_df=dh_df,
                    duals=duals,
                    filter_graph=filter_graph
                )
        
        spfa_time = time.time() - spfa_start
        spfa_times.append(spfa_time)
        tracker.record("spfa_time", spfa_time)
        
        mode = "(parallel)" if use_multiprocessing and len(source_nodes) > 1 else "(sequential)"
        print(f"  Total SPFA time {mode}: {spfa_time:.2f}s")
        
        # ========== PROCESS LABELS ==========
        print("\n► Processing labels...")
        label_start = time.time()
        
        # Process SPFA results and add columns
        filter_start = time.time()
        new_columns_added, reduced_cost_contribution = process_spfa_results(
            all_labels, k, last_route_number, S
        )
        filter_time = time.time() - filter_start
        
        total_reduced_cost += reduced_cost_contribution
        
        label_time = time.time() - label_start
        label_times.append(label_time)
        filter_times.append(filter_time)
        tracker.record("label_time", label_time)
        
        print(f"  Label processing time: {label_time:.2f}s")
        print(f"  Added {new_columns_added} new columns")
        
        # ========== CHECK FOR NEW COLUMNS ==========
        if new_columns_added == 0:
            if check_integrality(vars_values, config.INTEGRALITY_TOLERANCE):
                print("\n🥇🥇🥇 Solution is INTEGER AND OPTIMAL! 🥇🥇🥇")
                is_optimal = True
                break
            # Perform rounding
            fixed_vars = fix_vars(vars_values)
        
        # ========== UPDATE TRACKING ==========
        column_counts.append(len(S))
        iteration_time = time.time() - iteration_start
        iteration_times.append(iteration_time)
        iterations.append(f"{it:03d}")
        tracker.record("iteration_time", iteration_time)
        
        print(f"\n  Iteration {it} completed in {iteration_time:.2f}s")
        print(f"  Total columns: {len(S)}")
        
        it += 1
    
    # ========== FINALIZE RESULTS ==========
    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETED")
    print(f"{'='*80}")
    
    # Create results dataframe
    results_data = {
        'Iteration': iterations,
        'Experiment': experiment_name,
        'Time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Objective': [round(z, 2) for z in z_values],
        'Columns': column_counts,
        'Difference': [round(d, 2) for d in differences],
        'RMP_time': [round(t, 4) for t in rmp_times],
        'Graph_time': [round(t, 4) for t in graph_times],
        'SPFA_time': [round(t, 4) for t in spfa_times],
        'Label_time': [round(t, 4) for t in label_times],
        'Filter_time': [round(t, 4) for t in filter_times],
        'Iteration_time': [round(t, 4) for t in iteration_times]
    }
    
    results = create_results_dataframe(results_data)
    
    # Print summary
    print("\nResults Summary:")
    print(results[["Iteration", "Objective", "Columns", "Difference", "Iteration_time"]])
    print(f"\nFinal objective value: {min(z_values):.2f}")
    print(f"Total solving time: {sum(iteration_times):.2f} seconds")
    print(f"Total columns generated: {len(S) - initial_num_of_columns}")
    print(f"Final column count: {len(S)}")
    
    # Restore stdout
    sys.stdout = original_stdout
    logger.close()
    
    # Prepare metrics log
    generate_columns_end_time = time.time()
    generate_columns_runtime = generate_columns_end_time - generate_columns_start_time
    
    metrics = [
        "column_generation_time",
        "number_of_iterations",
        "start_objective",
        "final_objective",
        "is_optimal",
        "total_columns_generated",
        "total_reduced_cost"
    ]
    
    log_ids = [f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}{i}" 
               for i in range(len(metrics))]
    values = [
        generate_columns_runtime,
        len(iterations),
        z_values[0] if z_values else 0,
        min(z_values) if z_values else 0,
        is_optimal,
        len(S) - initial_num_of_columns,
        total_reduced_cost
    ]
    
    log_df = pd.DataFrame({
        "log_ids": log_ids,
        "experiment_name": [experiment_name] * len(metrics),
        "step": ["column_generation"] * len(metrics),
        "metric": metrics,
        "value": values,
    })
    
    # Create lightweight columns
    S_light = create_light_columns(S)
    
    # Extract non-zero columns
    nonzero_cols = extract_nonzero_columns(vars_values, var_to_route_mapping, S)
    
    return (S, min(z_values) if z_values else 0, z_values, column_counts, 
            results, experiment_name, log_df, S_light, nonzero_cols)


# ============================================================================
# MAIN BATCH EXECUTION FUNCTION
# ============================================================================

def main(tmax_values_per_set=None,
         base_instance_path=None,
         num_experiment_sets=None,
         k_values_per_set=None,
         filter_graph=None,
         z_min=None,
         max_iter=None,
         use_multiprocessing=True):
    """
    Main function to run the column generation algorithm in batch mode.
    
    This version creates multiple experiment sets with improved structure
    and performance optimizations.
    """
    
    # Use default values if not provided
    filter_graph = filter_graph if filter_graph is not None else FILTER_GRAPH
    
    # Validation
    if len(tmax_values_per_set) != len(k_values_per_set):
        raise ValueError(
            f"tmax_values_per_set and k_values_per_set must have the same length. "
            f"Got {len(tmax_values_per_set)} and {len(k_values_per_set)}"
        )
    
    # Print header
    print("\n" + "="*80)
    print("COLUMN GENERATION ALGORITHM - BATCH EXECUTION")
    print("="*80)
    print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Number of experiment sets: {num_experiment_sets}")
    print(f"Experiments per set: {len(k_values_per_set)}")
    print(f"K values per set: {k_values_per_set}")
    print(f"Tmax values per set: {tmax_values_per_set}")
    print(f"Total experiments: {num_experiment_sets * len(k_values_per_set)}")
    print(f"Multiprocessing: {use_multiprocessing}")
    print("="*80 + "\n")
    
    # Initialize base data
    dh_df, dh_times_df = initialize_data()
    
    # Track overall results
    total_start_time = time.time()
    all_sets_results = []
    
    # Process each experiment set
    for set_idx in range(num_experiment_sets):
        set_number = set_idx + 1
        exp_set_id = f"experiment_set_{set_number:03d}"
        exp_note = (f"Experiment set {set_number} with K values {k_values_per_set} "
                   f"and Tmax values {tmax_values_per_set}")
        
        print("\n" + "="*80)
        print(f"EXPERIMENT SET {set_number}/{num_experiment_sets}: {exp_set_id}")
        print("="*80)
        print(f"Notes: {exp_note}")
        
        # Create experiment set directory
        os.makedirs(f"experiments/{exp_set_id}", exist_ok=True)
        
        # Generate instance for this set
        print(f"\n► Generating new instance for set {set_number}...")
        
        gen, initial_solution_first, used_depots, instance_name, gen_time = generate_instance(
            tmax_values_per_set[0], None
        )
        
        instance_path_for_set = gen.timetables_path
        
        # Filter depots
        filtered_depots = {key: value for key, value in depots.items() 
                          if key in used_depots}
        
        # Build connection network
        graph, graph_build_time = build_connection_network(
            gen.timetables_path, used_depots
        )
        
        # Generate additional initial solutions
        initial_solutions = [initial_solution_first]
        initial_solution_gen_times = [0]
        
        for tmax_idx in range(1, len(tmax_values_per_set)):
            print(f"\n► Generating initial solution {tmax_idx + 1}/{len(tmax_values_per_set)} "
                  f"for set {set_number}")
            init_sol, init_gen_time = generate_initial_solution_with_tmax(
                gen, tmax_values_per_set[tmax_idx]
            )
            initial_solutions.append(init_sol)
            initial_solution_gen_times.append(init_gen_time)
        
        # Initialize set results
        set_results = pd.DataFrame()
        set_summary = {
            'set_number': set_number,
            'exp_set_id': exp_set_id,
            'instance_name': instance_name,
            'instance_path': instance_path_for_set,
            'experiments': []
        }
        
        # Run experiments in this set
        for exp_idx, (k_value, tmax_value) in enumerate(zip(k_values_per_set, tmax_values_per_set)):
            exp_number = exp_idx + 1
            
            print(f"\n► Running Experiment {exp_number}/{len(k_values_per_set)} in Set {set_number}")
            print(f"  Parameters: k={k_value}, tmax={tmax_value}, z_min={z_min}, "
                  f"max_iter={max_iter}, filter={filter_graph}")
            print(f"  Initial solution size: {len(initial_solutions[exp_idx])} routes")
            
            # Run column generation
            (columns, optimal, z_vals, col_counts, results, exp_name, 
             log_df, columns_light, nonzero_cols) = generate_columns(
                S=initial_solutions[exp_idx].copy(),
                graph=graph,
                depots=filtered_depots,
                dh_df=dh_df,
                dh_times_df=dh_times_df,
                z_min=z_min,
                k=k_value,
                max_iter=max_iter,
                experiment_name=f"exp_{exp_number:03d}_k{k_value}_tmax{tmax_value}",
                filter_graph=filter_graph,
                timetables_path=gen.timetables_path,
                exp_set_id=exp_set_id,
                instance_name=instance_name,
                use_multiprocessing=use_multiprocessing
            )
            
            # Track experiment results
            total_exp_time = (results['Iteration_time'].sum() + 
                            initial_solution_gen_times[exp_idx])
            
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
            
            # Add identifiers to results
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
            'Initial_Solution_Gen_Time': [e['initial_solution_gen_time'] 
                                         for e in set_summary['experiments']],
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
    
    # Final summary
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
    
    # Return last experiment's results for compatibility
    return columns, optimal, columns_light


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Example usage with default parameters
    main(
        tmax_values_per_set=[100, 150, 200],
        num_experiment_sets=2,
        k_values_per_set=[10, 20, 30],
        filter_graph=True,
        z_min=0.1,
        max_iter=10,
        use_multiprocessing=True
    )