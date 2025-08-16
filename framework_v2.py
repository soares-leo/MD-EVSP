from initializer.inputs import *
from initializer.generator import Generator
from initializer.utils import *
from initializer.conn_network_builder import GraphBuilder
from column_generator.utils import add_reduced_cost_info
from column_generator.rmp_solver import run_rmp
#from column_generator.spfa import run_spfa
from column_generator.spfa_v9 import run_spfa
#from genetic_algorithm.genetic_algorithm import run_ga
from collections import deque, namedtuple
from datetime import timedelta
import datetime
import json  # added for JSON output
import time
import numpy as np
import sys
import os
import json

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

# Init data
lines_info = lines_info
cp_depot_distances = cp_depot_distances
cp_locations_summary = summarize_cp_locations(lines_info)
cp_distances = calculate_cp_distances(cp_locations_summary, lines_info)
transformed_cp_depot_distances = transform_cp_depot_distances(cp_depot_distances)
dh_dict = merge_distances_dicts(transformed_cp_depot_distances, cp_distances)
dh_df = make_deadhead_df(dh_dict)
dh_times_df = make_deadhead_times_df(20, dh_df)

path_1 = None
path_alt = "initializer/files/instance_2025-08-16_00-10-31_l4_290_l59_100_l60_120.csv"

gen = Generator(lines_info, cp_depot_distances, depots, timetables_path_to_use=path_alt, seed=1)
initial_solution, used_depots, instance_name = gen.generate_initial_set()

depots = {key: value for key, value in depots.items() if key in used_depots}

# Connection Network
graph_builder = GraphBuilder(gen.timetables_path, used_depots)
graph = graph_builder.build_graph()

def generate_columns(S, graph, depots, dh_df, dh_times_df, Z_min, K, I, exp_number, filter_graph=True, timetables_path_to_use=gen.timetables_path, exp_set_id=None):
    timetables_name = gen.timetables_path.split("/")[-1]

    experiment_name = f"exp_{exp_number}_zmin_{Z_min}_k_{K}_i_{I}_{str(filter_graph)}"
    log_filename = f"experiments/{exp_set_id}/{experiment_name}/exp_{exp_number}.log"

    logger = Logger(log_filename)
    original_stdout = sys.stdout
    sys.stdout = logger

    cnt = 0
    it = 1
    it_list = []
    Z_values = []
    DIFFS = []
    lens_of_s = []
    durations = []
    spfa_runtimes = []
    label_corr_durations = []
    rmp_solve_durations = []
    red_cost_graph_durations = []
    cols_filtering_durations=[]

    while cnt <= I:
        
        iteration_start_time = time.time()

        it_len = len(str(it))
        zeros = 3 - it_len
        it_str = f"{zeros * '0'}{str(it)}"
        it_list.append(it_str)

        print()
        print("**************************************************************************")
        print("|                                                                        |")
        print(f"|                        I T E R A T I O N : {it_str}                         |")
        print("|                                                                        |")
        print("**************************************************************************")
        
        rmp_solve_start_time = time.time()
        
        status, model, current_cost, duals = run_rmp(
            depots,
            S,
            timetables_csv=timetables_path_to_use,
            cplex_path="C:/Program Files/IBM/ILOG/CPLEX_Studio2211/cplex/bin/x64_win64/cplex.exe"
        )

        rmp_solve_end_time = time.time()
        rmp_solve_duration = rmp_solve_end_time - rmp_solve_start_time
        rmp_solve_durations.append(rmp_solve_duration)

        with open(f"experiments/{exp_set_id}/{experiment_name}/RMP_for_iteration_{it}.txt", "w") as f:
            f.write(model.__str__())

        if it == 1:
            last_z = current_cost
        else:
            last_z = Z_values[-1]
        
        Z_values.append(current_cost)
        
        if abs(current_cost - last_z) < Z_min:
            cnt += 1
        else:
            cnt = 0

        DIFFS.append(abs(current_cost - last_z))

        # Reduced costs calculations
        red_cost_graph_start_time = time.time()
        red_cost_graph = add_reduced_cost_info(graph, duals, dh_times_df)
        red_cost_graph_end_time = time.time()
        red_cost_graph_duration = red_cost_graph_end_time - red_cost_graph_start_time
        red_cost_graph_durations.append(red_cost_graph_duration)


        # SPFA Run
        trip_keys = list(map(lambda x: int(x.split("_")[-1]), list(S.keys())))
           
        last_route_number = max(trip_keys)
        source_nodes = [n for n, d in graph.nodes(data=True) if d.get("type")=="K"]
        all_labels = {}
        times = []
        
        print()
        print("Current Z-values list:")
        print(Z_values)
        print()
        
        print()
        print("-"*50)
        print("SPFA RUN")
        print("-"*50)
        print()

        spfa_start_time = time.time()

        for i, t in enumerate(source_nodes):
            print(f"Running SPFA for source {t} ({i+1} of {len(source_nodes)})...")
            start_time = time.time()
            all_labels[t] = run_spfa(red_cost_graph, t, D=120, T_d=16*60, dh_df=dh_df, duals=duals, filter_graph=filter_graph) #, last_route_number=last_route_number, duals=duals)
            end_time = time.time()
            elapsed_time = end_time - start_time
            times.append(elapsed_time)
            print(f"SPFA for source {t} successfully finished! Duration: {elapsed_time:.2f}s.")
            print(f"Remaining time prediction: {np.mean(times) * (len(source_nodes) - (i+1)):.2f}s.")
            print(f"Total time prediction: {np.mean(times) * len(source_nodes):.2f}s.\n")

        spfa_end_time = time.time()
        spfa_runtime = spfa_end_time - spfa_start_time
        spfa_runtimes.append(spfa_runtime)

        print(f"SPFA was done in {np.sum(times):.2f}s.\n")

        label_corr_start_time = time.time()

        flat = {}
        for depot, routes in all_labels.items():
            for rkey, info in routes.items():
                if flat.get(rkey) is None:
                    flat[rkey] = []
                for data in info:
                    flat[rkey].append(data)

# ----- CORRECTED LOGIC FOR FILTERING ROUTES -----
        # This block safely filters routes by building a new list of survivors,
        # avoiding the bug of modifying a list while iterating over it.
        # Your custom filtering rule is preserved exactly as requested.

        new_flat = []
        for k, routes_in_group in flat.items():
            if not routes_in_group:
                continue

            routes_to_keep = []
            
            # For each route, we check if it should be kept or discarded
            for route_b in routes_in_group:
                is_discarded = False
                
                # Check if there is any other route 'a' that discards 'b'
                for route_a in routes_in_group:
                    if route_a is route_b:
                        continue

                    # Your rule: route 'b' is discarded if a route 'a' exists
                    # that has a lower/equal cost AND a higher/equal distance.
                    if route_a["ReducedCost"] <= route_b["ReducedCost"] and route_a["Data"]["total_travel_dist"] >= route_b["Data"]["total_travel_dist"]:
                        is_discarded = True
                        break # Found a reason to discard route_b, no need to check further

                # If after checking all other routes, none discarded it, then we keep it.
                if not is_discarded:
                    routes_to_keep.append(route_b)

            new_flat.extend(routes_to_keep)

        # To ensure the final list has no duplicate routes that might have survived the filter
        # we can create a unique list based on the path.

        label_corr_end_time = time.time()
        label_corr_duration = label_corr_end_time - label_corr_start_time
        label_corr_durations.append(label_corr_duration)

        cols_filtering_start_time = time.time()

        unique_final_routes = list({tuple(route['Path']): route for route in new_flat}.values())

        # Sort ascending by ReducedCost (most negative first)
        unique_final_routes.sort(key=lambda x: x["ReducedCost"])

        # Take the top K routes
        topk = unique_final_routes[:K]

        # ------------------- END OF CORRECTION --------------------

        # Build new entries, renaming keys to continue from last_route_number+1

        print()
        print("-"*50)
        print(f"TOP {K} GENERATED COLS (ROUTES)")
        print("-"*50)
        print()

        new_entries = {}
        for i, info in enumerate(topk, start=last_route_number+1):
            if info["ReducedCost"] >= 0:
                continue
            new_key = f"Route_{i}"
            new_entries[new_key] = {
                "Path": info["Path"],
                "Cost": info["Cost"],
                "Data": info["Data"]
            }
            print(f"{new_key}: ReducedCost = {info['ReducedCost']:.2f}, Path = {' → '.join(info['Path'])}")

        S.update(new_entries)
        lens_of_s.append(len(S))
        it+=1
        iteration_end_time = time.time()
        cols_filtering_duration = iteration_end_time - cols_filtering_start_time
        cols_filtering_durations.append(cols_filtering_duration)
        iteration_duration = iteration_end_time - iteration_start_time
        durations.append(iteration_duration)

    print("\n*********************  E N D   O F   T H E   L O O P  ********************")

    print("\nZ value evolution:")
    
    results = pd.DataFrame(
        {
            'Iteration': it_list,
            'Experiment': [experiment_name] * len(it_list),
            'Time': [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * len(it_list),
            'Objective': [round(_a, 2) for _a in Z_values],
            'Columns': lens_of_s,
            'Difference': [round(_b, 2) for _b in DIFFS],
            'RMP_time': [round(_c, 4) for _c in rmp_solve_durations],
            'Graph_time': [round(_d, 4) for _d in red_cost_graph_durations],
            'SPFA_time': [round(_e, 4) for _e in spfa_runtimes],
            'Label_time': [round(_f, 4) for _f in label_corr_durations],
            'Filter_time': [round(_g, 4) for _g in cols_filtering_durations],
            'Iteration_time': [round(_h, 4) for _h in durations]
        }
    )

    print(results[["Objective", "Columns", "Difference", "RMP_time", "Graph_time", "SPFA_time", "Label_time", "Filter_time", "Iteration_time"]])
    print()
    print("Total solving time:", sum(durations), "seconds")
    print()
    print("\nProcess is done! Results also in run_output.log.")

    sys.stdout = original_stdout
    logger.log.close()

    return S, min(Z_values), Z_values, lens_of_s, results


#exp_nums = [1,2,3,4,5,6,7,8,9,10,11,12]
exp_nums = [3]
#filter_options = [True, True, True, False, False, False, True, True, True, False, False, False]
filter_options = [True]
#Ks = [30,30,30,30,30,30,100,100,100,100,100,100]
Ks = [100]
exp_set_id = "comparision_new_instance_10_min_interval_v2"
#exp_set_id = "ga_test"
exp_set_ids = [exp_set_id] * len(exp_nums)
notes = """
    Rodagem com menos trips para comparacao com spfa v6.
    """
os.makedirs(f"experiments/{exp_set_id}", exist_ok=True)

if len(exp_nums) != len(filter_options) or len(exp_nums) != len(Ks):
    raise ValueError
else:
    merged = pd.DataFrame()

    for ex, op, k, id in zip(exp_nums, filter_options, Ks, exp_set_ids):
        columns, optimal, Z_values, lens_of_s, results = generate_columns(
            S=initial_solution.copy(),
            graph=graph,
            depots=depots,
            dh_df=dh_df,
            dh_times_df=dh_times_df,
            Z_min=50,
            K=k,
            I=9,
            exp_number=ex,
            filter_graph=op,
            exp_set_id=id,
            #timetables_path_to_use="initializer/files/instance_2025-08-03_18-47-46_l4_58_l59_20_l60_24.csv"
        )
        merged = pd.concat([merged, results], ignore_index=True)

    merged.to_csv(f"experiments/{exp_set_id}/results_report.csv", index=False)
    with open(f"experiments/{exp_set_id}/notes.txt", "w") as f:
        f.write(f"NOTES:\n{notes}\nInstance name: {instance_name}")


import pprint
import pandas as pd
import numpy as np

# Instead of saving to JSON, save the dictionary to a .py file
output_py_filename = f"experiments/{exp_set_id}/columns_data.py"

with open(output_py_filename, "w", encoding="utf-8") as f:
    # Add the necessary imports to the top of the file so it's self-contained
    f.write("import pandas as pd\n")
    f.write("import numpy as np\n\n")
    
    # Write the dictionary variable assignment
    f.write("columns = ")
    
    # Pretty-print the dictionary object directly into the file stream
    # This preserves the object types like pd.Timestamp and np.float64
    pprint.pprint(columns, stream=f, indent=4)

print(f"\n✅ Dictionary successfully saved with original types to: {output_py_filename}")

