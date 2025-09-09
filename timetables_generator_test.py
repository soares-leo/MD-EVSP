from initializer.generator_v2 import InstanceGenerator, SolutionGenerator
from initializer.inputs import *
from genetic_algorithm.genetic_algorithm_v4 import run_ga
import pandas as pd
import datetime
import os
from framework_v6 import generate_columns
from utils import initialize_data, build_connection_network

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

GA_PARAMS = [
    {
        "max_gen": 2000,
        "crossover_p": 0.3,
        "mutation_p": 0.01,
        "pop_size": 100,
        "elite_indivs": 5,
        "w1": 0.6,
        "w2": 0.6,
        "w3": 0.4,
        "w4": 1000,
        "gene_segments": 20
    },
    {
        "max_gen": 2000,
        "crossover_p": 0.3,
        "mutation_p": 0.01,
        "pop_size": 100,
        "elite_indivs": 5,
        "w1": 0.6,
        "w2": 1,
        "w3": 0.4,
        "w4": 1000,
        "gene_segments": 20
    },
    {
        "max_gen": 2000,
        "crossover_p": 0.3,
        "mutation_p": 0.01,
        "pop_size": 100,
        "elite_indivs": 5,
        "w1": 0.6,
        "w2": 2,
        "w3": 0.4,
        "w4": 1000,
        "gene_segments": 20
    }
]

NUM_OF_SETS = 1
TMAX_VALUES = [10.0, 15.0, 30.0]
K_VALUES = [50, 50, 50]
FILTERS = [False, False, False]
Z_MINS = [50, 50, 50]
MAX_ITERS = [2, 2, 2]
TYPES = ["GACG"]
SEEDS = [1, 1, 1]
INITIAL_SOLUTIONS_FOR_GA = [10,10,10]

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
    timetables = instance_gen.generate_instance()

    for type in TYPES:
        set_name = f"exp_set_{i+1:03d}_{type}"
        for _i, (tmax, k_val, opt_filter, z_min, max_iter, seed, ga_sols) in enumerate(
            zip(
                TMAX_VALUES,
                K_VALUES,
                FILTERS,
                Z_MINS,
                MAX_ITERS,
                SEEDS,
                INITIAL_SOLUTIONS_FOR_GA
            )
        ):
            
            exp_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
            experiment_name = f"{type}_{i}{_i}_k{k_val}_i{max_iter}_z{z_min}_tmax{tmax}_filter{opt_filter}_{exp_timestamp}"
            experiment_path = f"experiments/{set_name}/{experiment_name}"
            os.makedirs(experiment_path, exist_ok=True)

            if type == "GACG":
                
                all_solutions = []
                unique_routes = {}
                route_counter = 0
                used_depots = []

                for __i in range(ga_sols):
                    # IMPORTANT: Create a fresh copy of depots for each solution
                    depots_copy = {k: v.copy() for k, v in depots.items()}
                    
                    solution_gen = SolutionGenerator(
                        lines_info=lines_info, 
                        cp_depot_distances=cp_depot_distances, 
                        depots=depots_copy,  # Use the copy instead of depots.copy()
                        timetables_path_to_use=timetables[1], 
                        seed=seed, 
                        tmax=tmax
                    )
                    
                    # IMPORTANT: Reset the 'covered' status in timetables before each generation
                    # The SolutionGenerator reads the CSV fresh each time, but we need to ensure
                    # the CSV doesn't have leftover 'covered' markings
                    
                    solution, _used_depots, instance_name, gen_time = solution_gen.generate_initial_set()
                    
                    # Check if solution is valid (has routes)
                    if solution:
                        log_df = solution_gen.log_solution_generation(log_df, experiment_name)
                        all_solutions.append(solution)
                        used_depots.extend(_used_depots)
                    else:
                        print(f"Warning: Solution {__i+1} generated no routes")

                if not all_solutions:
                    print(f"Error: No valid solutions generated for experiment {experiment_name}")
                    continue

                print()
                print("Number of routes in each solution:")
                print([len(item) for item in all_solutions])
                print()

                used_depots = set(used_depots)
                
                filtered_depots = {key: value for key, value in depots.items() if key in used_depots}
            
                seen_routes = set()
                
                for solution in all_solutions:
                    for route_name, route_data in solution.items():
                        
                        route_tuple = tuple(route_data["Path"])
                        
                        if route_tuple not in seen_routes:
                            seen_routes.add(route_tuple)
                            unique_routes[f"Route_{route_counter}"] = route_data
                            route_counter += 1

                if not unique_routes:
                    print(f"Error: No unique routes found for experiment {experiment_name}")
                    continue
                
                print()
                print("Number of routes in unique solution:")
                print(len(unique_routes))
                print()

                best_fitness, best_cost, selected_columns, log_df = run_ga(unique_routes, timetables[1], experiment_name, log_df)

                # Read fresh timetables for graph building

                """
                fresh_timetables = pd.read_csv(timetables[1], converters={"departure_time": pd.to_datetime})
                graph, graph_log_df = build_connection_network(fresh_timetables, used_depots, experiment_name)

                log_df = pd.concat([log_df, graph_log_df], ignore_index=True)

                columns, optimal, z_vals, col_counts, results, exp_name, gencol_log_df = generate_columns(
                    S=selected_columns,
                    graph=graph,
                    depots=filtered_depots,
                    dh_df=dh_df,
                    dh_times_df=dh_times_df,
                    z_min=z_min,
                    k=k_val,
                    max_iter=max_iter,
                    experiment_name=experiment_name,
                    filter_graph=opt_filter,
                    timetables_path=timetables[1],
                    exp_set_id=set_name,
                    instance_name=instance_name
                )
                
                log_df = pd.concat([log_df, gencol_log_df], ignore_index=True)
                """

    log_df = instance_gen.log_instance_generation(log_df, experiment_name)

    

    print(log_df)