from initializer.inputs import *
from initializer.generator import Generator
from initializer.utils import *
from initializer.conn_network_builder import GraphBuilder
from column_generator.utils import add_reduced_cost_info
from column_generator.rmp_solver import run_rmp
#from column_generator.spfa import run_spfa
from column_generator.spfa import run_spfa
from genetic_algorithm.genetic_algorithm import run_ga
from collections import deque, namedtuple
from datetime import timedelta
import datetime
import json  # added for JSON output
import time
import numpy as np
import sys

class Logger(object):
    def __init__(self, filename="rmp_output.log"):
        self.terminal = sys.stdout
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

# Initial Solution
logger_0 = Logger('initial_solution.log')
original_stdout_0 = sys.stdout
sys.stdout = logger_0

gen = Generator(lines_info, cp_depot_distances, depots)
initial_solution, used_depots = gen.generate_initial_set()

sys.stdout = original_stdout_0
logger_0.log.close()

depots = {key: value for key, value in depots.items() if key in used_depots}

# Connection Network
graph_builder = GraphBuilder('initializer/files/timetables.csv', used_depots)
graph = graph_builder.build_graph()

def generate_columns(S, graph, depots, dh_df, dh_times_df, Z_min, K, I, log_filename="run_output.log"):

    logger = Logger(log_filename)
    original_stdout = sys.stdout
    sys.stdout = logger

    cnt = 0
    it = 1
    it_list = []
    Z_values = []
    lens_of_s = []
   

    while cnt <= I:

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
        status, model, current_cost, duals = run_rmp(
            depots,
            S,
            timetables_csv="initializer/files/timetables.csv",
            cplex_path="C:/Program Files/IBM/ILOG/CPLEX_Studio2211/cplex/bin/x64_win64/cplex.exe"
        )

        with open(f"RMP_for_iteration_{it}.txt", "w") as f:
            f.write(model.__str__())
        
        Z_values.append(current_cost)
        
        optimal = min(Z_values)
        
        if abs(current_cost - optimal) < Z_min:
            cnt += 1

        # Reduced costs calculations
        red_cost_graph = add_reduced_cost_info(graph, duals, dh_times_df)

        # SPFA Run
        trip_keys = list(map(lambda x: int(x.split("_")[-1]), list(S.keys())))
           
        last_route_number = max(trip_keys)
        source_nodes = [n for n, d in graph.nodes(data=True) if d.get("type")=="K"]
        all_labels = {}
        times = []
        
        print()
        print("-"*50)
        print("SPFA RUN")
        print("-"*50)
        print()

        for i, t in enumerate(source_nodes):
            print(f"Running SPFA for source {t} ({i+1} of {len(source_nodes)})...")
            start_time = time.time()
            all_labels[t] = run_spfa(red_cost_graph, t, D=120, T_d=16*60, dh_df=dh_df) #, last_route_number=last_route_number, duals=duals)
            end_time = time.time()
            elapsed_time = end_time - start_time
            times.append(elapsed_time)
            print(f"SPFA for source {t} successfully finished! Duration: {elapsed_time:.2f}s.")
            print(f"Remaining time prediction: {np.mean(times) * (len(source_nodes) - (i+1)):.2f}s.")
            print(f"Total time prediction: {np.mean(times) * len(source_nodes):.2f}s.\n")

        print(f"SPFA was done in {np.sum(times):.2f}s.\n")

        # Flatten all_labels into a list of (route_key, info)
        flat = []
        for depot, routes in all_labels.items():
            for rkey, info in routes.items():
                flat.append((rkey, info))

        # Sort ascending by ReducedCost (most negative first)
        flat.sort(key=lambda x: x[1]["ReducedCost"])

        # Take the top 10
        topk = flat[:K]

        # Build new entries, renaming keys to continue from last_route_number+1

        print()
        print("-"*50)
        print(f"TOP {K} GENERATED COLS (ROUTES)")
        print("-"*50)
        print()

        new_entries = {}
        for i, (old_key, info) in enumerate(topk, start=last_route_number+1):
            if info["ReducedCost"] >= 0:
                continue
            new_key = f"Route_{i}"
            new_entries[new_key] = {
                "Path": info["Path"],
                "Cost": info["Cost"],
                "Data": info["Data"]
            }
            print(f"{new_key} (was {old_key}): ReducedCost = {info['ReducedCost']:.2f}, Path = {' → '.join(info['Path'])}")

        S.update(new_entries)
        lens_of_s.append(len(S))
        it+=1

    print("\n*********************  E N D   O F   T H E   L O O P  ********************")

    print("\nZ value evolution:")
    for it, z, s in zip(it_list, Z_values, lens_of_s):
        print(f"Iteration {it} | Obj: {z:.4f} | Num of cols: {s}")
    
    
    print("\nProcess is done! Results also in run_output.log.")

    sys.stdout = original_stdout
    logger.log.close()

    return S, optimal, Z_values, lens_of_s

columns, optimal, Z_values, lens_of_s = generate_columns(
    S=initial_solution,
    graph=graph,
    depots=depots,
    dh_df=dh_df,
    dh_times_df=dh_times_df,
    Z_min=50,
    K=30,
    I=9
)

# final_solution, optimal_fitness = run_ga(
#     S=initial_solution,
#     Lmin=140,
#     Lmax=180,
#     pop_size=100,
#     max_gen=20000,
#     crossover_prob=0.3,
#     mutation_prob=0.01,
#     elite_size=5
# )

# print(optimal_fitness)

#156173.1259


# teste com mais linhas
# teste com depositos com menos capacidade
