from initializer.generator_v3 import SolutionGenerator
from initializer.inputs import *
from ..utils import initialize_data, build_connection_network
from framework_v7 import generate_columns
from .parameters import _3_POOL_GA_CG_GA_PARAMS as params # This one can stay as is
import pandas as pd
from genetic_algorithm.genetic_algorithm_static_diverse_double_pop import run_ga as run_ga_2
from genetic_algorithm.genetic_algorithm_static_diverse_pop import run_ga as run_ga_1
#from genetic_algorithm.genetic_algorithm_static_diverse_double_pop import run_ga as run_ga_2
import json
import datetime
import asyncio


def run_cg_ga(i):

    log_dfs = []

    # SINGLE INITIAL SOLUTION ===============================================================================================================================================================
    dh_df, dh_times_df = initialize_data()

    indep_depots_copy = {k: v.copy() for k, v in depots.items()}

    indep_solution_generator = SolutionGenerator(
        lines_info=lines_info,
        cp_depot_distances=cp_depot_distances,
        depots=indep_depots_copy,
        timetables_path_to_use=params["solution_generator"]["timetables_path"],
        seed=params["solution_generator"]["seed"],
        tmax=params["solution_generator"]["tmax"],
        random_start_trip=params["solution_generator"]["random_start_trip"],
        random_tmax=params["solution_generator"]["random_tmax"],
        tmax_min=params["solution_generator"]["tmax_min"],
        inverse=params["solution_generator"]["inverse"],
        experiment_name=i
    )

    indep_initial_solution_done_flag = False
    while not indep_initial_solution_done_flag:
        try:
            indep_initial_solution, indep_used_depots, instance_name, indep_solution_generation_time, indep_init_log_df = indep_solution_generator.generate_initial_set()
            indep_initial_solution_done_flag = True
        except Exception:
            continue
    
    log_dfs.append(indep_init_log_df)

    # POOL GENERATION ========================================================================================================================================================================

    pool_tmax_list = [16*60] * 6
    pool_random_st_list = ["random"] * 6
    pool_random_tmax_list = [True] * 3
    pool_random_tmax_list.extend([False] * 3)
    pool_inverse_list = [True, False] * 3
    pool_tmax_min_list = [0] * 3
    pool_tmax_min_list.extend([params["solution_generator"]["tmax"]] * 3)
    
    all_solutions = []
    used_depots = []
    
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
            except Exception:
                continue
        
        pool_log_df = pd.concat([pool_log_df, init_log_df], ignore_index=True)
                
        all_solutions.append(solution)
        used_depots.extend(_used_depots)

        INTERNAL_IT += 1
            
    # Deduplicate routes from pool
    start_time = datetime.datetime.now()
    unique_routes = {}
    route_counter = 0
    seen_routes = set()   
    for solution in all_solutions:
        for route_name, route_data in solution.items():
            route_tuple = tuple(route_data["Path"])
            if route_tuple not in seen_routes:
                seen_routes.add(route_tuple)
                unique_routes[f"Route_{route_counter}"] = route_data
                route_counter += 1
    duration = (datetime.datetime.now() - start_time).total_seconds()
    log_dfs.append(pd.DataFrame([{
        "log_ids": f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}{i}",
        "experiment_name": i,
        "step": "preprocessing_and_transformations",
        "metric": "pool_deduplication_run_time",
        "value": duration
    }]))

    # Compose single log df with both pool and independent solution metrics
    solution_generation_time = pool_log_df[pool_log_df.metric == "solution_generation_time"].value.sum()
    total_generated_columns = len(unique_routes)

    metrics = [
        "pool_solution_generation_time",
        "pool_total_generated_columns"
    ]

    log_ids = [f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}{i}" for i in range(len(metrics))]
    values = [solution_generation_time, total_generated_columns]
    summary_pool_log_df = pd.DataFrame({
        "log_ids": log_ids,
        "experiment_name": [i] * len(metrics),
        "step": ["initial_solution"] * len(metrics),
        "metric": metrics,
        "value": values,
    })
    log_dfs.append(summary_pool_log_df)

    # FIRST GA ========================================================================================================================================================================
    
    best_fitness, best_cost, v, r, u, s, selected_columns, first_ga_log_df = run_ga_1(
        unique_routes,
        timetables_path=params["solution_generator"]["timetables_path"],
        experiment_name=i,
        ga_params=params["FIRST_GA"]
    )
    log_dfs.append(first_ga_log_df)

    # Combine and deduplicate solutions
    start_time = datetime.datetime.now()
    combined_selected_columns = selected_columns.copy()
    max_route_num = max([int(r.split('_')[1]) for r in selected_columns.keys()] + [-1])
    for route_name, route_data in indep_initial_solution.items():
        max_route_num += 1
        new_route_name = f"Route_{max_route_num}"
        combined_selected_columns[new_route_name] = route_data

    _unique_routes = {}
    _route_counter = 0
    _seen_routes = set()   
    for route_name, route_data in combined_selected_columns.items():
        route_tuple = tuple(route_data["Path"])
        if route_tuple not in _seen_routes:
            _seen_routes.add(route_tuple)
            _unique_routes[f"Route_{_route_counter}"] = route_data
            _route_counter += 1
    duration = (datetime.datetime.now() - start_time).total_seconds()
    log_dfs.append(pd.DataFrame([{
        "log_ids": f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}{i}",
        "experiment_name": i,
        "step": "preprocessing_and_transformations",
        "metric": "combination_deduplication_run_time",
        "value": duration
    }]))
    
    # GRAPH BUILD ========================================================================================================================================================================
    
    used_depots.extend(indep_used_depots)
    used_depots = set(used_depots)
    filtered_depots = {key: value for key, value in depots.items() if key in used_depots}

    fresh_timetables = pd.read_csv(
        params["solution_generator"]["timetables_path"],
        converters={"departure_time": pd.to_datetime}
    )

    graph, graph_log_df = build_connection_network(fresh_timetables, used_depots, i)
    log_dfs.append(graph_log_df)
    
    # COLUMN GENERATION ========================================================================================================================================================================

    columns, optimal, z_vals, col_counts, results, exp_name, gencol_log_df, columns_light = generate_columns(
        S=_unique_routes,
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
    log_dfs.append(gencol_log_df)

    # Dedup final columns
    # start_time = datetime.datetime.now()
    # final_unique_routes = {}
    # final_route_counter = 0
    # final_seen_routes = set()   
    # for start_trip, columns in columns.items():
    #     for route_name, route_data in col.items():
    #         route_tuple = tuple(route_data["Path"])
    #         if route_tuple not in final_seen_routes:
    #             final_seen_routes.add(route_tuple)
    #             final_unique_routes[f"Route_{final_route_counter}"] = route_data
    #             final_route_counter += 1
    # duration = (datetime.datetime.now() - start_time).total_seconds()
    # log_dfs.append(pd.DataFrame([{
    #     "log_ids": f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}{i}",
    #     "experiment_name": i,
    #     "step": "preprocessing_and_transformations",
    #     "metric": "final_deduplication_run_time",
    #     "value": duration
    # }]))

    # SECOND GA ========================================================================================================================================================================

    sec_best_fitness, sec_best_cost, sec_v, sec_r, sec_u, sec_s, sec_selected_columns, sec_ga_log_df = run_ga_2(
        columns,
        timetables_path=params["solution_generator"]["timetables_path"],
        experiment_name=i,
        ga_params=params["SECOND_GA"]
    )
    log_dfs.append(sec_ga_log_df)

    final_df = pd.concat(log_dfs, ignore_index=True)

    print(final_df)

    final_df.to_csv(f"experiments/3_pool_GA_CG_GA/reports/3_pool_GA_CG_GA_{i}.csv", index=False)

    if sec_selected_columns:
        file_path = f"experiments/3_pool_GA_CG_GA/solutions/3_pool_GA_CG_GA_{i}.json"
        with open(file_path, 'w') as f:
            json.dump(sec_selected_columns, f, indent=2, default=str)
        print(f"Selected columns saved to: {file_path}")
    

if __name__ == "__main__":
    for tmax in [10]:#, 15, 20, 30, 60*8]:
        
        params["solution_generator"]["tmax"] = tmax

        for i in range(1,6):
        
            run_cg_ga(str(i) + "_pop200")