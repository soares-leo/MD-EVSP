from initializer.generator_v2 import InstanceGenerator, SolutionGenerator
from initializer.inputs import *
from genetic_algorithm.genetic_algorithm_static import run_ga
import pandas as pd
import datetime
import pickle
import os
from framework_v7 import generate_columns
from utils import initialize_data, build_connection_network
from ga_params import GA_PARAMS

# Configuration constants
TRIPS_CONFIGS = [
    {
        "line": 4,
        "total_trips": 290,
        "first_start_time": "05:50",
        "last_start_time": "22:50",
    },
    {
        "line": 59,
        "total_trips": 100,
        "first_start_time": "06:40",
        "last_start_time": "19:50",
    },
    {
        "line": 60,
        "total_trips": 120,
        "first_start_time": "06:00",
        "last_start_time": "21:10",
    },
]

NUM_OF_SETS = 1
TMAX_SETS = [[16*60, 16*60, 16*60, 16*60, 16*60, 16*60]]
RANDOM_START_TRIP_SETS = [["random", "random", "random", "random", "random", "random"]]
RANDOM_TMAX_SETS = [[True, True, False, True, True, False]]
TMAX_MIN_SETS = [[0, 0, 15, 0, 0, 15]]
INVERSE_SETS = [[True, False, False, True, False, False]]
TMAX_VALUES = [10.0, 15.0, 30.0]
K_VALUES = [50, 50, 50]
FILTERS = [False, False, False]
Z_MINS = [50, 50, 50]
MAX_ITERS = [2, 2, 2]
TYPES = ["GACG"]
SEEDS = [None, None, None]
INITIAL_SOLUTIONS_FOR_GA = [10, 10, 10]

# Configuration for the independent initial solution
INDEPENDENT_SOLUTION_CONFIG = {
    "tmax": 15,
    "random_start_trip": "no",
    "random_tmax": False,
    "tmax_min": 0,
    "inverse": False,
    "seed": None
}

EXPERIMENT_SETS_CONFIGS = {
    "sets": NUM_OF_SETS,
    "experiments": {
        "types": {
            type: {
                "k_values": K_VALUES,
                "tmax_values": TMAX_VALUES,
                "filters": FILTERS,
                "z_mins": Z_MINS,
                "max_iters": MAX_ITERS,
                "seeds": SEEDS
            } for type in TYPES
        },
    }
}


def save_columns_to_file(columns_data, file_path):
    """
    Save column generation results to a pickle file.
    
    Args:
        columns_data: Dictionary containing columns and metadata
        file_path: Path to save the file
    """
    with open(file_path, 'wb') as f:
        pickle.dump(columns_data, f)
    print(f"✓ Columns saved to {file_path}")


def run_experiments():
    """Main function that runs experiments up to column generation."""
    
    # Initialize data once at the beginning
    dh_df, dh_times_df = initialize_data()
    
    # Create directory for saved columns
    os.makedirs("saved_columns", exist_ok=True)
    
    for i in range(NUM_OF_SETS):
        
        log_df = pd.DataFrame(columns=["experiment_name", "step", "metric", "value"])
        instance_gen = InstanceGenerator(lines_info=lines_info, trips_configs=TRIPS_CONFIGS)
        timetables = 'generated_timetables/combined_timetable_20250903_225333.csv'
        
        for type in TYPES:
            set_name = f"exp_set_k500_{i+1:03d}_{type}"
            
            for tmax_list, rand_ear, rand_tmax, tmax_min_list, inverse_list in zip(
                TMAX_SETS, RANDOM_START_TRIP_SETS, RANDOM_TMAX_SETS, 
                TMAX_MIN_SETS, INVERSE_SETS
            ):
                
                # STEP 1: Generate independent initial solution
                print("=" * 80)
                print("STEP 1: Generating independent initial solution...")
                print("=" * 80)
                
                depots_copy_independent = {k: v.copy() for k, v in depots.items()}
                
                solution_gen_independent = SolutionGenerator(
                    lines_info=lines_info,
                    cp_depot_distances=cp_depot_distances,
                    depots=depots_copy_independent,
                    timetables_path_to_use=timetables,
                    seed=INDEPENDENT_SOLUTION_CONFIG["seed"],
                    tmax=INDEPENDENT_SOLUTION_CONFIG["tmax"],
                    random_start_trip=INDEPENDENT_SOLUTION_CONFIG["random_start_trip"],
                    random_tmax=INDEPENDENT_SOLUTION_CONFIG["random_tmax"],
                    tmax_min=INDEPENDENT_SOLUTION_CONFIG["tmax_min"],
                    inverse=INDEPENDENT_SOLUTION_CONFIG["inverse"]
                )
                
                independent_solution, independent_used_depots, _, _ = solution_gen_independent.generate_initial_set()
                
                if not independent_solution:
                    print("Warning: Independent solution generated no routes")
                    independent_solution = {}
                
                print(f"✓ Independent solution generated with {len(independent_solution)} routes")
                print()
                
                # STEP 2: Generate solutions for GA
                print("=" * 80)
                print("STEP 2: Generating solutions for GA...")
                print("=" * 80)
                
                all_solutions = []
                unique_routes = {}
                route_counter = 0
                used_depots = []
                
                for __i in range(len(tmax_list)):
                    print(f"Generating solution {__i + 1}/{len(tmax_list)}...")
                    
                    depots_copy = {k: v.copy() for k, v in depots.items()}
                    
                    solution_gen = SolutionGenerator(
                        lines_info=lines_info,
                        cp_depot_distances=cp_depot_distances,
                        depots=depots_copy,
                        timetables_path_to_use=timetables,
                        seed=None,
                        tmax=tmax_list[__i],
                        random_start_trip=rand_ear[__i],
                        random_tmax=rand_tmax[__i],
                        tmax_min=tmax_min_list[__i],
                        inverse=inverse_list[__i]
                    )
                    
                    solution, _used_depots, instance_name, gen_time = solution_gen.generate_initial_set()
                    
                    if solution:
                        all_solutions.append(solution)
                        used_depots.extend(_used_depots)
                        print(f"  ✓ Generated {len(solution)} routes")
                    else:
                        print(f"  ⚠ Warning: Solution generated no routes")
                
                if not all_solutions:
                    print(f"Error: No valid solutions generated for experiment")
                    continue
                
                print()
                print("Number of routes in each solution for GA:")
                for idx, item in enumerate(all_solutions):
                    print(f"  Solution {idx + 1}: {len(item)} routes")
                print()
                
                # Combine used_depots
                used_depots.extend(independent_used_depots)
                used_depots = set(used_depots)
                
                filtered_depots = {key: value for key, value in depots.items() if key in used_depots}
                
                # Deduplicate routes for GA
                print("Deduplicating routes for GA...")
                seen_routes = set()
                
                for solution in all_solutions:
                    for route_name, route_data in solution.items():
                        route_tuple = tuple(route_data["Path"])
                        
                        if route_tuple not in seen_routes:
                            seen_routes.add(route_tuple)
                            unique_routes[f"Route_{route_counter}"] = route_data
                            route_counter += 1
                
                if not unique_routes:
                    print(f"Error: No unique routes found for experiment")
                    continue
                
                print(f"✓ Number of unique routes for GA: {len(unique_routes)}")
                print()
                
                for _i, params in enumerate(GA_PARAMS):
                    
                    if type == "GACG":
                        
                        # STEP 3: Run GA (Pre-Column Generation)
                        print("=" * 80)
                        print(f"STEP 3: Running GA (Pre-CG) with parameter set {_i + 1}/{len(GA_PARAMS)}...")
                        print("=" * 80)
                        
                        try:
                            best_fitness, best_cost, selected_columns, _ = run_ga(
                                unique_routes, timetables, _i, log_df, tmax_list, params
                            )
                            print(f"✓ Pre-CG GA completed. Best cost: {best_cost}, Selected columns: {len(selected_columns)}")
                            
                        except Exception as e:
                            print(f"⚠ Error in Pre-CG GA: {e}")
                            continue
                        
                        # STEP 4: Combine solutions
                        print()
                        print("=" * 80)
                        print("STEP 4: Combining solutions...")
                        print("=" * 80)
                        print(f"Concatenating independent solution ({len(independent_solution)} routes) " +
                              f"to selected_columns ({len(selected_columns)} routes)")
                        
                        combined_selected_columns = selected_columns.copy()
                        
                        max_route_num = max([int(r.split('_')[1]) for r in selected_columns.keys()] + [-1])
                        
                        for route_name, route_data in independent_solution.items():
                            max_route_num += 1
                            new_route_name = f"Route_{max_route_num}"
                            combined_selected_columns[new_route_name] = route_data
                        
                        print(f"✓ Combined solution has {len(combined_selected_columns)} routes total")
                        
                        # STEP 5: Column Generation
                        print()
                        print("=" * 80)
                        print("STEP 5: Running Column Generation...")
                        print("=" * 80)
                        
                        fresh_timetables = pd.read_csv(timetables, converters={"departure_time": pd.to_datetime})
                        graph, graph_log_df = build_connection_network(fresh_timetables, used_depots, _i)
                        
                        print(f"Starting column generation with parameters:")
                        print(f"  - Initial columns: {len(combined_selected_columns)}")
                        print(f"  - z_min: 1000")
                        print(f"  - k: 100")
                        print(f"  - max_iter: 19")
                        print(f"  - filter_graph: False")
                        print()
                        
                        columns, optimal, z_vals, col_counts, results, exp_name, gencol_log_df = generate_columns(
                            S=combined_selected_columns,
                            graph=graph,
                            depots=filtered_depots,
                            dh_df=dh_df,
                            dh_times_df=dh_times_df,
                            z_min=400,
                            k=100,
                            max_iter=19,
                            experiment_name=_i,
                            filter_graph=False,
                            timetables_path=timetables,
                            exp_set_id=set_name,
                            instance_name=instance_name
                        )
                        
                        print()
                        print("=" * 80)
                        print("COLUMN GENERATION COMPLETED")
                        print("=" * 80)
                        print(f"✓ Final optimal value: {optimal}")
                        print(f"✓ Total columns generated: {len(columns)}")
                        print(f"✓ Iterations: {len(z_vals)}")
                        print()
                        
                        # SAVE COLUMNS FOR LATER USE
                        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                        columns_file = f"saved_columns/columns_set_{i+1}_params_{_i+1}_{timestamp}.pkl"
                        
                        columns_data = {
                            'columns': columns,
                            'metadata': {
                                'set_number': i + 1,
                                'param_set': _i + 1,
                                'timetables_path': timetables,
                                'optimal_value': optimal,
                                'num_columns': len(columns),
                                'timestamp': timestamp,
                                'pre_cg_ga_params': params,
                                'column_generation_params': {
                                    'z_min': 1000,
                                    'k': 100,
                                    'max_iter': 19,
                                    'filter_graph': False
                                }
                            }
                        }
                        
                        save_columns_to_file(columns_data, columns_file)
                        
                        print("=" * 80)
                        print(f"COLUMNS SAVED TO: {columns_file}")
                        print("=" * 80)
                        print("\nYou can now use test_post_cg_ga.py to test GA parameters on these columns.")
                        print()
    
    print()
    print("=" * 80)
    print("COLUMN GENERATION PHASE COMPLETED")
    print("=" * 80)
    print("\nGenerated column files are saved in 'saved_columns/' directory")
    print("Use test_post_cg_ga.py to run post-CG GA experiments")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("STARTING COLUMN GENERATION ONLY ORCHESTRATOR")
    print("=" * 80)
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Number of experiment sets: {NUM_OF_SETS}")
    print(f"Types: {TYPES}")
    print("=" * 80 + "\n")
    
    run_experiments()
    
    print("\n✓ Column generation phase completed successfully!")
    print("✓ Check 'saved_columns/' directory for generated column files")