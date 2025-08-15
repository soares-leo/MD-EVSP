from initializer.inputs import *
from initializer.generator import Generator
from initializer.utils import *
from initializer.conn_network_builder import GraphBuilder
from column_generator.utils import add_reduced_cost_info
from column_generator.rmp_solver import run_rmp
from column_generator.spfa_trip_centric import run_spfa_for_paths
from datetime import timedelta
import datetime
import time
import numpy as np
import sys
import os
import pandas as pd
import pprint
import copy

# --- Index Constants for Label Tuple ---
NODE_IDX, TOTAL_WEIGHT_IDX, TOTAL_DIST_IDX, TOTAL_TIME_IDX, DIST_SINCE_CHARGE_IDX, \
TIME_SINCE_CHARGE_IDX, CURRENT_TIME_IDX, PATH_IDX, TOTAL_DH_DIST_IDX, TOTAL_DH_TIME_IDX, \
TOTAL_TRAVEL_DIST_IDX, TOTAL_TRAVEL_TIME_IDX, TOTAL_WAIT_TIME_IDX, TOTAL_CHARGING_TIME_IDX = range(14)

class Logger(object):
    def __init__(self, filename="rmp_output.log"):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# --- Global Data Setup ---
lines_info = lines_info
cp_depot_distances = cp_depot_distances
cp_locations_summary = summarize_cp_locations(lines_info)
cp_distances = calculate_cp_distances(cp_locations_summary, lines_info)
transformed_cp_depot_distances = transform_cp_depot_distances(cp_depot_distances)
dh_dict = merge_distances_dicts(transformed_cp_depot_distances, cp_distances)
dh_df = make_deadhead_df(dh_dict)
dh_times_df = make_deadhead_times_df(20, dh_df)

def generate_columns(S, graph, depots, dh_df, dh_times_df, Z_min, K, I, exp_number, exp_set_id=None, timetables_path_to_use=None):
    experiment_name = f"exp_{exp_number}_zmin_{Z_min}_k_{K}_i_{I}"
    log_filename = f"experiments/{exp_set_id}/{experiment_name}/exp_{exp_number}.log"
    logger = Logger(log_filename)
    original_stdout = sys.stdout
    sys.stdout = logger

    cnt, it, Z_values, last_z, log_data = 0, 1, [], float('inf'), []
    print("Instance file path:", timetables_path_to_use)
    
    while True:
        iteration_start_time = time.time()
        print(f"\n{'*'*70}\n|{'I T E R A T I O N : %03d' % it:^68}|\n{'*'*70}")

        rmp_solve_start_time = time.time()
        status, model, current_cost, duals = run_rmp(
            depots, S, timetables_csv=timetables_path_to_use,
            cplex_path="C:/Program Files/IBM/ILOG/CPLEX_Studio2211/cplex/bin/x64_win64/cplex.exe"
        )
        rmp_solve_duration = time.time() - rmp_solve_start_time

        Z_values.append(current_cost)
        diff = abs(current_cost - last_z)
        cnt = cnt + 1 if it > 1 and diff < Z_min else 0
        last_z = current_cost
        print(f"\nConvergence Check: Objective diff = {diff:.2f}, Stop counter = {cnt}/{I}")
        print("Current Z-values list:", [f"{z:.2f}" for z in Z_values])

        log_entry = {'Iteration': it, 'Experiment': experiment_name, 'Time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     'Objective': round(current_cost, 2), 'Columns': len(S), 'Difference': round(diff, 2),
                     'RMP_time': round(rmp_solve_duration, 4)}

        if cnt >= I:
            print("\nStopping criterion met. Final RMP solved. Terminating loop.")
            log_entry.update({'Graph_time': 0, 'SPFA_time': 0, 'Filter_time': 0, 'Iteration_time': round(time.time() - iteration_start_time, 4)})
            log_data.append(log_entry)
            break

        red_cost_graph_start_time = time.time()
        red_cost_graph = add_reduced_cost_info(graph, duals, dh_times_df)
        red_cost_graph_duration = time.time() - red_cost_graph_start_time
        
        print("\n--- Solving Subproblems (Path Extraction Phase) ---")
        all_labels_from_all_depots = {}
        source_nodes = [n for n, d in red_cost_graph.nodes(data=True) if d.get("type") == "K"]
        common_nodes = [node for node, data in red_cost_graph.nodes(data=True) if data.get("type") in ["T", "C"]]
        
        spfa_start_time = time.time()
        for source_depot in source_nodes:
            nodes_for_subgraph = common_nodes + [source_depot]
            depot_subgraph = red_cost_graph.subgraph(nodes_for_subgraph)
            all_labels_from_all_depots[source_depot] = run_spfa_for_paths(
                depot_subgraph, source_depot, D=120, T_d=60*16, dh_df=dh_df
            )
        spfa_duration = time.time() - spfa_start_time
        print(f"Path extraction phase completed in {spfa_duration:.2f}s.")

        print("\n--- Re-organizing paths by first trip and creating full columns ---")
        paths_per_trip = {}
        for start_depot, labels_dict in all_labels_from_all_depots.items():
            for end_node, labels in labels_dict.items():
                for label in labels:
                    path = label[PATH_IDX]
                    if len(path) < 2 or not path[1].startswith('l'): continue
                    
                    first_trip = path[1]
                    if first_trip not in paths_per_trip: paths_per_trip[first_trip] = []
                    
                    for end_depot in source_nodes:
                        if red_cost_graph.has_edge(end_node, end_depot):
                            arc_to_depot = red_cost_graph[end_node][end_depot]
                            if (label[TOTAL_TIME_IDX] + arc_to_depot['time'] > (60*16) or 
                                label[DIST_SINCE_CHARGE_IDX] + arc_to_depot['dist'] > 120): continue
                            
                            final_path = list(path) + [end_depot]
                            recharging_freq = sum(1 for node in final_path if node.startswith("c"))
                            final_dh_time = label[TOTAL_DH_TIME_IDX] + arc_to_depot['time']
                            cost = calculate_cost(recharging_freq, label[TOTAL_TRAVEL_TIME_IDX], final_dh_time, label[TOTAL_WAIT_TIME_IDX])
                            trips_in_path = [node for node in final_path if graph.nodes[node]['type'] == 'T']
                            alpha_sum = sum(duals['alpha'][trip] for trip in trips_in_path)
                            beta_val = duals['beta'][start_depot]
                            reduced_cost = cost - alpha_sum - beta_val
                            
                            paths_per_trip[first_trip].append({
                                "Path": final_path, "Cost": cost, "ReducedCost": reduced_cost,
                                "TotalDistance": label[TOTAL_DIST_IDX] + arc_to_depot['dist'],
                                "Data": { "total_dh_time": final_dh_time, "total_travel_time": label[TOTAL_TRAVEL_TIME_IDX],
                                          "total_wait_time": label[TOTAL_WAIT_TIME_IDX], "total_charging_time": label[TOTAL_CHARGING_TIME_IDX] }
                            })
        
        print("\n--- Applying Post-SPFA Dominance Rule (per trip) ---")
        post_spfa_start_time = time.time()
        surviving_columns = []
        for first_trip, group in paths_per_trip.items():
            non_dominated_in_group = []
            for col_a in group:
                is_dominated = False
                for col_b in group:
                    if col_a is col_b: continue
                    if (col_b['ReducedCost'] <= col_a['ReducedCost'] and col_b['TotalDistance'] >= col_a['TotalDistance']):
                        is_dominated = True
                        break
                if not is_dominated:
                    non_dominated_in_group.append(col_a)
            surviving_columns.extend(non_dominated_in_group)
        post_spfa_duration = time.time() - post_spfa_start_time
        print(f"Dominance rule applied in {post_spfa_duration:.2f}s. {len(surviving_columns)} columns survived.")

        cols_filtering_start_time = time.time()
        unique_new_columns = list({tuple(route['Path']): route for route in surviving_columns}.values())
        unique_new_columns.sort(key=lambda x: x["ReducedCost"])
        top_k_cols = [col for col in unique_new_columns if col['ReducedCost'] < -1e-6][:K]
        
        print(f"\n--- Adding Top {len(top_k_cols)} Columns (Max K={K}) ---")
        if not top_k_cols: print("No new columns with negative reduced cost found.")
        
        last_route_number = max([int(k.split('_')[-1]) for k in S.keys()] + [0])
        for i, info in enumerate(top_k_cols, start=last_route_number + 1):
            S[f"Route_{i}"] = {"Path": info["Path"], "Cost": info["Cost"], "Data": info["Data"]}
            print(f"  + Adding Route_{i}: ReducedCost = {info['ReducedCost']:.2f}, Path = {' -> '.join(info['Path'])}")
        cols_filtering_duration = time.time() - cols_filtering_start_time
        
        log_entry.update({'Graph_time': round(red_cost_graph_duration, 4), 'SPFA_time': round(spfa_duration + post_spfa_duration, 4), 
                          'Filter_time': round(cols_filtering_duration, 4), 'Iteration_time': round(time.time() - iteration_start_time, 4)})
        log_data.append(log_entry)
        it += 1

    print("\n" + "*"*21 + " E N D  O F  T H E  L O O P " + "*"*22)
    results_df = pd.DataFrame(log_data)
    print("\nZ value evolution:\n", results_df[["Objective", "Columns", "Difference", "RMP_time", "SPFA_time", "Iteration_time"]])
    print(f"\nTotal solving time: {results_df['Iteration_time'].sum():.2f} seconds")
    print("\nProcess is done! Results also in log file.")
    sys.stdout = original_stdout
    logger.log.close()
    return S, min(Z_values) if Z_values else float('inf'), results_df

if __name__ == "__main__":
    #experiment_names = [f"framework_v3_random_gen_{i:02d}" for i in range(1, 51)]
    # experiment_names = [
    #     f"instance_A_revisited_10K",
    #     f"instance_A_revisited_30K",
    #     f"instance_A_revisited_50K",
    #     f"instance_B_revisited_10K",
    #     f"instance_B_revisited_30K",
    #     f"instance_B_revisited_50K",
    #     f"instance_C_revisited_10K",
    #     f"instance_C_revisited_30K",
    #     f"instance_C_revisited_50K",
    #     f"instance_D_revisited_10K",
    #     f"instance_D_revisited_30K",
    #     f"instance_D_revisited_50K",
    #     f"instance_E_revisited_10K",
    #     f"instance_E_revisited_30K",
    #     f"instance_E_revisited_50K"
    # ]
    # Ks = [10, 30, 50, 10, 30, 50, 10, 30, 50, 10, 30, 50, 10, 30, 50]

    # instances = [
    #     "initializer/files/timetables_A.csv",
    #     "initializer/files/timetables_A.csv",
    #     "initializer/files/timetables_A.csv",
    #     "initializer/files/timetables_B.csv",
    #     "initializer/files/timetables_B.csv",
    #     "initializer/files/timetables_B.csv",
    #     "initializer/files/timetables_C.csv",
    #     "initializer/files/timetables_C.csv",
    #     "initializer/files/timetables_C.csv",
    #     "initializer/files/timetables_D.csv",
    #     "initializer/files/timetables_D.csv",
    #     "initializer/files/timetables_D.csv",
    #     "initializer/files/timetables_E.csv",
    #     "initializer/files/timetables_E.csv",
    #     "initializer/files/timetables_E.csv"
    # ]

    experiment_names = [
        f"reduced_dominance_K30_240m_2025-08-07_01-42-34_l4_290_l59_100_l60_120_revisited",
        f"reduced_dominance_K50_240m_2025-08-07_01-42-34_l4_290_l59_100_l60_120_revisited"
    ]

    Ks = [30, 50]

    instances = [
        "initializer/files/instance_2025-08-07_01-42-34_l4_290_l59_100_l60_120.csv",
        "initializer/files/instance_2025-08-07_01-42-34_l4_290_l59_100_l60_120.csv"
    ]

    for i, (exp_name, k, inst) in enumerate(zip(experiment_names, Ks, instances)):
        print("\n" + "="*80)
        print(f"S T A R T I N G   E X P E R I M E N T   {i} / {len(experiment_names)}: {exp_name}")
        print("="*80 + "\n")
        print()
        print("Description: Teste do novo metodo aplicado a uma instancia resolvida com o v2 (instance_2025-08-07_01-42-34_l4_290_l59_100_l60_120)")

        current_depots = copy.deepcopy(depots)
        gen = Generator(lines_info, cp_depot_distances, current_depots, timetables_path_to_use=inst, seed=1)
        initial_solution, used_depots, instance_name = gen.generate_initial_set()
        instance_depots = {key: value for key, value in current_depots.items() if key in used_depots}
        graph_builder = GraphBuilder(gen.timetables_path, used_depots)
        instance_graph = graph_builder.build_graph()

        exp_set_id = f"{exp_name}_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        os.makedirs(f"experiments/{exp_set_id}", exist_ok=True)

        final_columns, optimal_cost, results_df = generate_columns(
            S=initial_solution.copy(), graph=instance_graph, depots=instance_depots, 
            dh_df=dh_df, dh_times_df=dh_times_df,
            Z_min=50, K=k, I=10, exp_number=i, exp_set_id=exp_set_id,
            timetables_path_to_use=gen.timetables_path
        )

        results_df.to_csv(f"experiments/{exp_set_id}/results_report.csv", index=False)
        output_py_filename = f"experiments/{exp_set_id}/columns_data.py"
        with open(output_py_filename, "w", encoding="utf-8") as f:
            f.write("import pandas as pd\nimport numpy as np\n\ncolumns = ")
            pprint.pprint(final_columns, stream=f, indent=4)
        print(f"\n✅ Final columns and results successfully saved to: experiments/{exp_set_id}/")

