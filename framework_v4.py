# %% Imports
import logging
import os
import sys
import time
from datetime import datetime
from itertools import chain
from pathlib import Path
import pprint

import numpy as np
import pandas as pd

# Local module imports (preserved as requested)
from column_generator.rmp_solver import run_rmp
from column_generator.spfa_v9 import run_spfa
from column_generator.utils import add_reduced_cost_info
from initializer.conn_network_builder import GraphBuilder
from initializer.generator import Generator
from initializer.inputs import *
from initializer.utils import (
    calculate_cp_distances,
    make_deadhead_df,
    make_deadhead_times_df,
    merge_distances_dicts,
    summarize_cp_locations,
    transform_cp_depot_distances,
)

# %% Configuration
# --- Experiment Set Configuration ---
EXP_SET_ID = "comparision_new_instance_10_min_interval_v4"
EXPERIMENTS = [
    # (exp_num, filter_graph_option, K_value)
    (3, False, 100),
    # You can add more experiment configurations here
    # (4, True, 100),
    # (5, False, 150),
]
NOTES = """
    Rodagem com menos trips para comparacao com spfa v6.
    Refactored for performance and clarity.
"""

# --- Column Generation Parameters ---
Z_MIN_CONVERGENCE = 50.0  # Threshold for convergence
MAX_STALL_ITERATIONS = 9  # Max iterations (I) without improvement before stopping
CPLEX_PATH = "C:/Program Files/IBM/ILOG/CPLEX_Studio2211/cplex/bin/x64_win64/cplex.exe"
SPFA_D_PARAM = 120        # Max driving time per shift in minutes
SPFA_TD_PARAM = 16 * 60   # Max total duration per shift in minutes


# %% Helper Classes and Functions

class PerformanceTimer:
    """A context manager to time and log the duration of a code block."""
    def __init__(self, name: str, logger: logging.Logger, metrics_list: list):
        self.name = name
        self.logger = logger
        self.metrics_list = metrics_list
        self.start_time = None

    def __enter__(self):
        self.logger.info(f"Starting: {self.name}...")
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start_time
        self.metrics_list.append(duration)
        self.logger.info(f"Finished: {self.name}. Duration: {duration:.4f}s")


def setup_logging(log_path: Path) -> logging.Logger:
    """Configures logging to both file and console."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # File handler
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def load_and_prepare_data(timetables_path_alt: str) -> dict:
    """Loads and preprocesses all initial data for the model."""
    logging.info("Starting initial data loading and preparation...")
    
    # Init data (assuming these are loaded from initializer.inputs)
    lines_info_data = lines_info
    cp_depot_distances_data = cp_depot_distances
    depots_data = depots

    # Process data
    cp_locations_summary = summarize_cp_locations(lines_info_data)
    cp_distances = calculate_cp_distances(cp_locations_summary, lines_info_data)
    transformed_cp_depot_distances = transform_cp_depot_distances(cp_depot_distances_data)
    dh_dict = merge_distances_dicts(transformed_cp_depot_distances, cp_distances)
    dh_df = make_deadhead_df(dh_dict)
    dh_times_df = make_deadhead_times_df(20, dh_df)

    # Generate initial solution
    gen = Generator(lines_info_data, cp_depot_distances_data, depots_data, timetables_path_to_use=timetables_path_alt, seed=1)
    initial_solution, used_depots, instance_name = gen.generate_initial_set()
    
    active_depots = {key: value for key, value in depots_data.items() if key in used_depots}

    # Build connection network graph
    graph_builder = GraphBuilder(gen.timetables_path, used_depots)
    graph = graph_builder.build_graph()

    logging.info("Data loading and preparation complete.")
    
    return {
        "graph": graph,
        "initial_solution": initial_solution,
        "depots": active_depots,
        "dh_df": dh_df,
        "dh_times_df": dh_times_df,
        "instance_name": instance_name,
        "timetables_path": gen.timetables_path
    }


def save_columns_to_py(columns: dict, output_path: Path):
    """Saves the final columns dictionary to a self-contained .py file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("import pandas as pd\n")
        f.write("import numpy as np\n\n")
        f.write("columns = ")
        pprint.pprint(columns, stream=f, indent=4)
    logging.info(f"✅ Dictionary successfully saved with original types to: {output_path}")


# %% Core Column Generation Logic

def _solve_rmp(depots, S, timetables_path, logger, metrics_list):
    """Solves the Restricted Master Problem and returns results."""
    with PerformanceTimer("RMP Solving", logger, metrics_list):
        status, model, cost, duals = run_rmp(
            depots, S, timetables_csv=timetables_path, cplex_path=CPLEX_PATH
        )
    # The model object can be large, returning its string representation might be better
    # if it needs to be logged, but here we just need cost and duals.
    return cost, duals, model.__str__()


def _run_pricing_subproblem(graph, duals, dh_df, dh_times_df, filter_graph_flag, logger, metrics_list):
    """Runs the SPFA for each source node to generate new columns."""
    all_labels = {}
    with PerformanceTimer("Total SPFA Run", logger, metrics_list):
        red_cost_graph = add_reduced_cost_info(graph, duals, dh_times_df)
        source_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "K"]
        
        individual_times = []
        for i, source_node in enumerate(source_nodes, 1):
            start_time = time.perf_counter()
            logger.info(f"Running SPFA for source {source_node} ({i} of {len(source_nodes)})...")
            
            labels = run_spfa(
                red_cost_graph, source_node, D=SPFA_D_PARAM, T_d=SPFA_TD_PARAM,
                dh_df=dh_df, duals=duals, filter_graph=filter_graph_flag
            )
            all_labels[source_node] = labels
            
            elapsed_time = time.perf_counter() - start_time
            individual_times.append(elapsed_time)
            mean_time = np.mean(individual_times)
            logger.info(f"SPFA for source {source_node} finished. Duration: {elapsed_time:.2f}s.")
            logger.info(f"  > Avg time/source: {mean_time:.2f}s. Est. remaining: {mean_time * (len(source_nodes) - i):.2f}s.")
            
    return all_labels


def _filter_dominated_routes(routes_in_group: list) -> list:
    """Filters out dominated routes within a group. Preserves original logic."""
    if not routes_in_group:
        return []

    routes_to_keep = []
    for route_b in routes_in_group:
        is_dominated = False
        for route_a in routes_in_group:
            if route_a is route_b:
                continue
            # A dominates B if A's cost is lower/equal AND A's distance is higher/equal
            if (route_a["ReducedCost"] <= route_b["ReducedCost"] and
                route_a["Data"]["total_travel_dist"] >= route_b["Data"]["total_travel_dist"]):
                is_dominated = True
                break
        if not is_dominated:
            routes_to_keep.append(route_b)
            
    return routes_to_keep


def _filter_and_select_columns(all_labels: dict, K: int, logger: logging.Logger, metrics_lists: dict):
    """Filters, sorts, and selects the top-K new columns to add."""
    # Flatten structure: group all generated routes by their destination node
    with PerformanceTimer("Label Correction/Grouping", logger, metrics_lists['label']):
        grouped_routes = {}
        
        # --- THIS IS THE CORRECTED LINE ---
        # It now iterates one level deeper to get the individual route dictionaries.
        all_routes = chain.from_iterable(
            dest_list for source_dict in all_labels.values() for dest_list in source_dict.values()
        )

        for route in all_routes:
            # Assuming the last node in the path is the key
            # This line will now work correctly because 'route' is a dictionary
            key = route['Path'][-1]
            if key not in grouped_routes:
                grouped_routes[key] = []
            grouped_routes[key].append(route)
            
    # Filter dominated routes within each group
    with PerformanceTimer("Columns Filtering", logger, metrics_lists['filter']):
        filtered_routes = []
        for routes_in_group in grouped_routes.values():
            # The custom filtering logic is applied here
            survivors = _filter_dominated_routes(routes_in_group)
            filtered_routes.extend(survivors)

        # Ensure uniqueness based on the path and select top K
        unique_routes = list({tuple(route['Path']): route for route in filtered_routes}.values())
        unique_routes.sort(key=lambda x: x["ReducedCost"])
        
        top_k_cols = [r for r in unique_routes[:K] if r["ReducedCost"] < 0]

    return top_k_cols


def generate_columns(S_initial: dict, graph, depots: dict, dh_df, dh_times_df, Z_min: float, K: int, I: int, exp_details: dict) -> tuple:
    """
    Main column generation loop.

    Args:
        S_initial: The initial set of routes (columns).
        graph: The connection network graph.
        ... (other data inputs)
        Z_min: Convergence threshold for the objective function.
        K: The number of new columns to add per iteration.
        I: Maximum number of iterations without improvement.
        exp_details: Dictionary with experiment metadata.

    Returns:
        A tuple containing the final solution, optimal value, and results dataframe.
    """
    # --- Setup ---
    exp_path = Path(f"experiments/{exp_details['set_id']}/{exp_details['name']}")
    logger = setup_logging(exp_path / f"exp_{exp_details['num']}.log")

    S = S_initial.copy()
    
    # --- Metrics Tracking ---
    metrics = {
        'iteration': [], 'objective': [], 'num_columns': [], 'difference': [],
        'total_time': [], 'rmp_time': [], 'graph_time': [], 'spfa_time': [],
        'label_time': [], 'filter_time': []
    }
    
    stall_counter = 0
    iteration = 1

    # --- Main Loop ---
    while stall_counter <= I:
        iter_start_time = time.perf_counter()
        it_str = f"{iteration:03d}"
        metrics['iteration'].append(it_str)
        
        logger.info("="*80)
        logger.info(f"|                     I T E R A T I O N : {it_str}                     |")
        logger.info("="*80)

        # 1. Solve Restricted Master Problem
        current_cost, duals, rmp_model_str = _solve_rmp(
            depots, S, exp_details['timetables_path'], logger, metrics['rmp_time']
        )
        (exp_path / f"RMP_iteration_{it_str}.txt").write_text(rmp_model_str)
        
        # 2. Check for convergence
        last_z = metrics['objective'][-1] if metrics['objective'] else current_cost
        diff = abs(current_cost - last_z)
        metrics['objective'].append(current_cost)
        metrics['difference'].append(diff)
        
        logger.info(f"Current Objective (Z): {current_cost:.2f}, Change from last: {diff:.2f}")

        if diff < Z_min:
            stall_counter += 1
            logger.info(f"Objective change is less than Z_min ({Z_min}). Stall count: {stall_counter}/{I+1}")
        else:
            stall_counter = 0
        
        if stall_counter > I:
            logger.info("Convergence criteria met. Exiting loop.")
            break
            
        # 3. Run Pricing Subproblem (SPFA) to generate new columns
        # The graph update time is implicitly part of the SPFA run timer, preserving original metric structure.
        all_labels = _run_pricing_subproblem(
            graph, duals, dh_df, dh_times_df, exp_details['filter_graph'], logger, metrics['spfa_time']
        )
        
        # 4. Filter, Sort, and Select Top-K New Columns
        top_k_new_cols = _filter_and_select_columns(
            all_labels, K, logger, {'label': metrics['label_time'], 'filter': metrics['filter_time']}
        )
        
        if not top_k_new_cols:
            logger.warning("No new columns with negative reduced cost found. This indicates optimality. Exiting loop.")
            break

        # 5. Add new columns to the problem
        logger.info(f"--- Top {len(top_k_new_cols)} selected new columns ---")
        last_route_number = max(int(k.split("_")[-1]) for k in S.keys())
        new_entries = {}
        for i, info in enumerate(top_k_new_cols, start=last_route_number + 1):
            new_key = f"Route_{i}"
            new_entries[new_key] = {"Path": info["Path"], "Cost": info["Cost"], "Data": info["Data"]}
            logger.info(f"  > Adding {new_key}: ReducedCost = {info['ReducedCost']:.2f}, Path = {' → '.join(info['Path'])}")
        
        S.update(new_entries)
        metrics['num_columns'].append(len(S))
        
        # --- Iteration Wrap-up ---
        iteration += 1
        metrics['total_time'].append(time.perf_counter() - iter_start_time)
        logger.info(f"Iteration {it_str} complete. Total columns: {len(S)}")

    logger.info("="*80)
    logger.info("              E N D   O F   T H E   L O O P")
    logger.info("="*80)
    
    # --- Final Results ---
    results_df = pd.DataFrame({
        'Iteration': metrics['iteration'],
        'Experiment': [exp_details['name']] * len(metrics['iteration']),
        'Time': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * len(metrics['iteration']),
        'Objective': [round(v, 2) for v in metrics['objective']],
        'Columns': metrics['num_columns'],
        'Difference': [round(v, 2) for v in metrics['difference']],
        'RMP_time': [round(v, 4) for v in metrics['rmp_time']],
        # 'Graph_time' is now part of SPFA time, can be added back if measured separately
        'SPFA_time': [round(v, 4) for v in metrics['spfa_time']],
        'Label_time': [round(v, 4) for v in metrics['label_time']],
        'Filter_time': [round(v, 4) for v in metrics['filter_time']],
        'Iteration_time': [round(v, 4) for v in metrics['total_time']]
    })

    logger.info("\n--- Z value evolution ---\n")
    logger.info(results_df[['Objective', 'Columns', 'Difference', 'RMP_time', 'SPFA_time', 'Iteration_time']])
    logger.info(f"\nTotal solving time: {sum(metrics['total_time']):.2f} seconds")
    
    optimal_value = min(metrics['objective']) if metrics['objective'] else float('inf')
    return S, optimal_value, results_df


def main():
    """Main script execution function."""
    # This path is an example from the original code
    timetables_path_alt = None
    
    # Load data once for all experiments
    data = load_and_prepare_data(timetables_path_alt)
    
    all_results = []
    for exp_num, filter_opt, k_val in EXPERIMENTS:
        
        experiment_details = {
            "num": exp_num,
            "set_id": EXP_SET_ID,
            "name": f"exp_{exp_num}_zmin_{int(Z_MIN_CONVERGENCE)}_k_{k_val}_i_{MAX_STALL_ITERATIONS}_{str(filter_opt)}",
            "filter_graph": filter_opt,
            "timetables_path": data['timetables_path']
        }
        
        # The main function now logs internally, but we can log the start of an experiment here
        logging.info(f"\n\n{'='*30} STARTING EXPERIMENT: {experiment_details['name']} {'='*30}\n")
        
        final_columns, optimal_value, results_df = generate_columns(
            S_initial=data['initial_solution'],
            graph=data['graph'],
            depots=data['depots'],
            dh_df=data['dh_df'],
            dh_times_df=data['dh_times_df'],
            Z_min=Z_MIN_CONVERGENCE,
            K=k_val,
            I=MAX_STALL_ITERATIONS,
            exp_details=experiment_details,
        )
        
        all_results.append(results_df)

        # Save final columns for this specific experiment
        output_py_path = Path(f"experiments/{EXP_SET_ID}/{experiment_details['name']}/final_columns.py")
        save_columns_to_py(final_columns, output_py_path)

    # Consolidate and save results from all experiments
    if all_results:
        merged_results = pd.concat(all_results, ignore_index=True)
        results_path = Path(f"experiments/{EXP_SET_ID}/results_report.csv")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        merged_results.to_csv(results_path, index=False)
        logging.info(f"Consolidated results report saved to: {results_path}")

    # Save experiment notes
    notes_path = Path(f"experiments/{EXP_SET_ID}/notes.txt")
    notes_path.write_text(f"NOTES:\n{NOTES}\nInstance name: {data['instance_name']}")
    logging.info(f"Experiment notes saved to: {notes_path}")


if __name__ == "__main__":
    # Setup a basic logger for the data loading part before experiment-specific logging takes over
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    main()