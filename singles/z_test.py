from initializer.generator_v2 import InstanceGenerator, SolutionGenerator
from initializer.inputs import *
from genetic_algorithm.genetic_algorithm_mixed import run_ga
import pandas as pd
import datetime
import os
from framework_v6 import generate_columns
from utils import initialize_data, build_connection_network
from ga_params import GA_PARAMS

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
TMAX_SETS = [[16*60, 16*60]]
RANDOM_START_TRIP_SETS = [["random", "random"]]
RANDOM_TMAX_SETS = [[True, True]]
TMAX_MIN_SETS = [[0, 0]]
INVERSE_SETS = [[True, False]]
TMAX_VALUES = [10.0, 15.0, 30.0]
K_VALUES = [500, 500, 500]
FILTERS = [False, False, False]
Z_MINS = [50, 50, 50]
MAX_ITERS = [2, 2, 2]
TYPES = ["GACG"]
SEEDS = [None, None, None]
INITIAL_SOLUTIONS_FOR_GA = [10,10,10]

# Configuration for the independent initial solution
INDEPENDENT_SOLUTION_CONFIG = {
    "tmax": 10,  # You can adjust this value
    "random_start_trip": "no",  # or specific value
    "random_tmax": False,
    "tmax_min": 0,
    "inverse": False,  # You can adjust this
    "seed": None
}

EXPERIMENT_SETS_CONFIGS = {
    "sets": NUM_OF_SETS, #for each set, a new instance will be generated
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

dh_df, dh_times_df = initialize_data()

for i in range(NUM_OF_SETS):

    log_df = pd.DataFrame(columns=["experiment_name", "step", "metric", "value"])
    instance_gen = InstanceGenerator(lines_info=lines_info, trips_configs=TRIPS_CONFIGS)
    timetables = 'initializer/files/instance_20250826203245271323_l4_l4_290_l59_100_l60_120.csv'

    for type in TYPES:
        set_name = f"exp_set_k500_{i+1:03d}_{type}"
        
        for tmax_list, rand_ear, rand_tmax, tmax_min_list, inverse_list  in zip(TMAX_SETS, RANDOM_START_TRIP_SETS, RANDOM_TMAX_SETS, TMAX_MIN_SETS, INVERSE_SETS):
            
            # STEP 1: Generate independent initial solution (not for GA)
            print("Generating independent initial solution...")
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
            
            print(f"Independent solution generated with {len(independent_solution)} routes")
            
            # STEP 2: Generate solutions for GA
            all_solutions = []
            unique_routes = {}
            route_counter = 0
            used_depots = []

            for __i in range(len(tmax_list)):
                # IMPORTANT: Create a fresh copy of depots for each solution
                depots_copy = {k: v.copy() for k, v in depots.items()}
                
                solution_gen = SolutionGenerator(
                    lines_info=lines_info, 
                    cp_depot_distances=cp_depot_distances, 
                    depots=depots_copy,  # Use the copy instead of depots.copy()
                    timetables_path_to_use=timetables, 
                    seed=None, 
                    tmax=tmax_list[__i],
                    random_start_trip=rand_ear[__i],
                    random_tmax=rand_tmax[__i],
                    tmax_min=tmax_min_list[__i],
                    inverse=inverse_list[__i]
                )
                
                solution, _used_depots, instance_name, gen_time = solution_gen.generate_initial_set()
                
                # Check if solution is valid (has routes)
                if solution:
                    log_df = solution_gen.log_solution_generation(log_df, "a")
                    all_solutions.append(solution)
                    used_depots.extend(_used_depots)
                else:
                    print(f"Warning: Solution generated no routes")

            if not all_solutions:
                print(f"Error: No valid solutions generated for experiment a")
                continue

            print()
            print("Number of routes in each solution for GA:")
            print([len(item) for item in all_solutions])
            print()

            # Combine used_depots from both GA solutions and independent solution
            used_depots.extend(independent_used_depots)
            used_depots = set(used_depots)
            
            filtered_depots = {key: value for key, value in depots.items() if key in used_depots}
        
            # Deduplicate routes for GA
            seen_routes = set()
            
            for solution in all_solutions:
                for route_name, route_data in solution.items():
                    
                    route_tuple = tuple(route_data["Path"])
                    
                    if route_tuple not in seen_routes:
                        seen_routes.add(route_tuple)
                        unique_routes[f"Route_{route_counter}"] = route_data
                        route_counter += 1

            if not unique_routes:
                print(f"Error: No unique routes found for experiment a")
                continue
            
            print()
            print("Number of routes in unique solution for GA:")
            print(len(unique_routes))
            print()        
            
            for _i, params in enumerate(GA_PARAMS):

                if type == "GACG":
                    
                    # STEP 3: Run GA
                    try:
                        best_fitness, best_cost, selected_columns, log_df = run_ga(unique_routes, timetables, _i, log_df, tmax_list, params)
                    except:
                        with open("error.txt", "a") as err:
                            err.write("an error occured when trying these parameters:")
                            err.write(str(params))
                            err.write("\n")
                        continue
                    
                    # STEP 4: Concatenate independent solution to selected_columns
                    print(f"\nConcatenating independent solution ({len(independent_solution)} routes) to selected_columns ({len(selected_columns)} routes)")
                    
                    # Create a combined solution dictionary
                    combined_selected_columns = selected_columns.copy()
                    
                    # Add routes from independent solution with unique names
                    max_route_num = max([int(r.split('_')[1]) for r in selected_columns.keys()] + [-1])
                    
                    for route_name, route_data in independent_solution.items():
                        max_route_num += 1
                        new_route_name = f"Route_{max_route_num}"
                        combined_selected_columns[new_route_name] = route_data
                    
                    print(f"Combined solution has {len(combined_selected_columns)} routes total")
                    
                    # STEP 5: Continue with column generation using combined solution
                    fresh_timetables = pd.read_csv(timetables, converters={"departure_time": pd.to_datetime})
                    graph, graph_log_df = build_connection_network(fresh_timetables, used_depots, _i)

                    log_df = pd.concat([log_df, graph_log_df], ignore_index=True)

                    columns, optimal, z_vals, col_counts, results, exp_name, gencol_log_df = generate_columns(
                        S=combined_selected_columns,  # Use the combined solution here
                        graph=graph,
                        depots=filtered_depots,
                        dh_df=dh_df,
                        dh_times_df=dh_times_df,
                        z_min=100,
                        k=500,
                        max_iter=9,
                        experiment_name=_i,
                        filter_graph=False,
                        timetables_path=timetables,
                        exp_set_id=set_name,
                        instance_name=instance_name
                    )
                    
                    log_df = pd.concat([log_df, gencol_log_df], ignore_index=True)

        log_df = instance_gen.log_instance_generation(log_df, _i)

    print(log_df)