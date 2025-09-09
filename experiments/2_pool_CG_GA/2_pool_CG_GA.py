from initializer.generator_v3 import SolutionGenerator
from initializer.inputs import *
from ..utils import initialize_data, build_connection_network
from framework_v7 import generate_columns
from .parameters import _1_POOL_CG_GA_PARAMS as params # This one can stay as is
import pandas as pd
from genetic_algorithm.genetic_algorithm_standard_v2 import run_ga
import json
import datetime


def run_cg_ga(i):

# SINGLE INITIAL SOLUTION
    dh_df, dh_times_df = initialize_data()

    # indep_depots_copy = {k: v.copy() for k, v in depots.items()}

    # indep_solution_generator = SolutionGenerator(
    #     lines_info=lines_info,
    #     cp_depot_distances=cp_depot_distances,
    #     depots=indep_depots_copy,
    #     timetables_path_to_use=params["solution_generator"]["timetables_path"],
    #     seed=params["solution_generator"]["seed"],
    #     tmax=params["solution_generator"]["tmax"],
    #     random_start_trip=params["solution_generator"]["random_start_trip"],
    #     random_tmax=params["solution_generator"]["random_tmax"],
    #     tmax_min=params["solution_generator"]["tmax_min"],
    #     inverse=params["solution_generator"]["inverse"],
    #     experiment_name=i
    # )

    # indep_initial_solution_done_flag = False
    # while not indep_initial_solution_done_flag:
    #     try:
    #         indep_initial_solution, indep_used_depots, instance_name, indep_solution_generation_time, indep_init_log_df = indep_solution_generator.generate_initial_set()
    #         indep_initial_solution_done_flag = True
    #     except:
    #         continue

    pool_tmax_list = [16*60] * 12
    pool_random_st_list = ["random"] * 12
    pool_random_tmax_list = [True] * 6
    pool_random_tmax_list.extend([False] * 6)
    pool_inverse_list = [True, False] * 6
    pool_tmax_min_list = [0] * 6
    pool_tmax_min_list.extend([params["solution_generator"]["tmax"]] * 6)
    
    all_solutions = []
    used_depots = []
    unique_routes = {}
    route_counter = 0

    pool_log_df = pd.DataFrame()

    INTERNAL_IT = 1

    for _tmax, _random_st, _random_tmax, _tmax_min, _inverse in zip(pool_tmax_list, pool_random_st_list, pool_random_tmax_list, pool_tmax_min_list, pool_inverse_list):
        print(INTERNAL_IT)
        depots_copy = {k: v.copy() for k, v in depots.items()}
        solution_generator = SolutionGenerator(
            lines_info=lines_info,
            cp_depot_distances=cp_depot_distances,
            depots=depots_copy,
            timetables_path_to_use=params["solution_generator"]["timetables_path"],
            seed=params["solution_generator"]["seed"],
            tmax=_tmax,
            random_start_trip=_random_st,
            random_tmax=_random_tmax,
            tmax_min=_tmax_min,
            inverse=_inverse,
            experiment_name=i
        )

        initial_solution_done_flag = False
        while not initial_solution_done_flag:
            try:
                solution, _used_depots, instance_name, solution_generation_time, init_log_df = solution_generator.generate_initial_set()
                initial_solution_done_flag = True
            except:
                continue
        
        pool_log_df = pd.concat([pool_log_df, init_log_df], ignore_index=True)
                   
        all_solutions.append(solution)
        used_depots.extend(_used_depots)

        # Combine used_depots
        #used_depots.extend(indep_used_depots)
        INTERNAL_IT += 1
                
    used_depots = set(used_depots)
    filtered_depots = {key: value for key, value in depots.items() if key in used_depots}
           
    # Deduplicate routes
    seen_routes = set()
            
    for solution in all_solutions:
        for route_name, route_data in solution.items():
            route_tuple = tuple(route_data["Path"])
            if route_tuple not in seen_routes:
                seen_routes.add(route_tuple)
                unique_routes[f"Route_{route_counter}"] = route_data
                route_counter += 1    

    fresh_timetables = pd.read_csv(
        params["solution_generator"]["timetables_path"],
        converters={"departure_time": pd.to_datetime}
    )

    solution_generation_time = pool_log_df[pool_log_df.metric == "solution_generation_time"].value.sum()
    total_generated_columns = len(unique_routes)

    metrics = [
        "solution_generation_time",
        "total_generated_columns"
    ]

    log_ids = [f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}{i}" for i in range(len(metrics))]
    values = [solution_generation_time, total_generated_columns]
    pool_log_df = pd.DataFrame({
        "log_ids": log_ids,
        "experiment_name": [i] * len(metrics),
        "step": ["initial_solution"] * len(metrics),
        "metric": metrics,
        "value": values,
    })

    print(fresh_timetables)

    graph, graph_log_df = build_connection_network(fresh_timetables, used_depots, i)
    
    columns, optimal, z_vals, col_counts, results, exp_name, gencol_log_df, columns_light = generate_columns(
        S=unique_routes,
        graph=graph,
        depots=filtered_depots,
        dh_df=dh_df,
        dh_times_df=dh_times_df,
        z_min=params["CG"]["z_min"],
        k=params["CG"]["k"],
        max_iter=params["CG"]["max_iter"],
        experiment_name=i,
        filter_graph=params["CG"]["filter_graph"],
        timetables_path=params["solution_generator"]["timetables_path"],
        exp_set_id="CG_GA",
        instance_name=instance_name
    )

    best_fitness, best_cost, v, r, u, s, selected_columns, ga_log_df = run_ga(
        columns,
        timetables_path=params["solution_generator"]["timetables_path"],
        experiment_name=i,
        ga_params=params["GA"]
    )

    final_df = pd.concat([pool_log_df, graph_log_df, gencol_log_df, ga_log_df], ignore_index=True)

    print(final_df)

    final_df.to_csv(f"experiments/2_pool_CG_GA/reports/2_pool_CG_GA_{i}.csv", index=False)

    if selected_columns:
        with open(f"experiments/2_pool_CG_GA/solutions/2_pool_CG_GA_{i}.json", 'w') as f:
            json.dump(selected_columns, f, indent=2, default=str)
        print(f"Selected columns saved to: selected_columns_{i}.json")
    

if __name__ == "__main__":
    for tmax in [10]:#, 15, 20, 30, 60*8]:
        
        params["solution_generator"]["tmax"] = tmax

        for i in range(1,6):
        
            run_cg_ga(str(i) + "_tmax" + str(tmax))
