from initializer.generator_v3 import SolutionGenerator
from initializer.inputs import *
from ..utils import initialize_data, build_connection_network
from framework_v7 import generate_columns
from .parameters import _1_CG_GA_PARAMS as params # This one can stay as is
import pandas as pd
from genetic_algorithm.genetic_algorithm_standard_v3 import run_ga
import json


def run_cg_ga(i):

# SINGLE INITIAL SOLUTION
    dh_df, dh_times_df = initialize_data()

    depots_copy = {k: v.copy() for k, v in depots.items()}

    solution_generator = SolutionGenerator(
        lines_info=lines_info,
        cp_depot_distances=cp_depot_distances,
        depots=depots_copy,
        timetables_path_to_use=params["solution_generator"]["timetables_path"],
        seed=params["solution_generator"]["seed"],
        tmax=params["solution_generator"]["tmax"],
        random_start_trip=params["solution_generator"]["random_start_trip"],
        random_tmax=params["solution_generator"]["random_tmax"],
        tmax_min=params["solution_generator"]["tmax_min"],
        inverse=params["solution_generator"]["inverse"],
        experiment_name=i
    )

    initial_solution_done_flag = False
    while not initial_solution_done_flag:
        try:
            initial_solution, used_depots, instance_name, solution_generation_time, init_log_df = solution_generator.generate_initial_set()
            initial_solution_done_flag = True
        except:
            continue

    fresh_timetables = pd.read_csv(
        params["solution_generator"]["timetables_path"],
        converters={"departure_time": pd.to_datetime}
    )

    print(fresh_timetables)

    graph, graph_log_df = build_connection_network(fresh_timetables, used_depots, i)
    
    columns, optimal, z_vals, col_counts, results, exp_name, gencol_log_df, columns_light = generate_columns(
        S=initial_solution,
        graph=graph,
        depots=depots,
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

    final_df = pd.concat([init_log_df, graph_log_df, gencol_log_df, ga_log_df], ignore_index=True)

    print(final_df)

    final_df.to_csv(f"experiments/1_CG_GA/reports/1_CG_GA_extra_{i}.csv", index=False)

    if selected_columns:
        with open(f"experiments/1_CG_GA/solutions/1_CG_GA_extra_{i}.json", 'w') as f:
            json.dump(selected_columns, f, indent=2, default=str)
        print(f"Selected columns saved to: selected_columns_{i}.json")
    

if __name__ == "__main__":
    for tmax in [15]:#, 15, 20, 30, 60*8]:
        
        params["solution_generator"]["tmax"] = tmax

        run_cg_ga(str(7) + "_tmax" + str(tmax))
