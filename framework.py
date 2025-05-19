from initializer.inputs import *
from initializer.generator import Generator
from initializer.utils import *
from initializer.conn_network_builder import GraphBuilder
from column_generator.utils import add_reduced_cost_info
from column_generator.rmp_solver import run_rmp
from column_generator.spfa import run_spfa
from genetic_algorithm.genetic_algorithm import run_ga
from collections import deque, namedtuple
from datetime import timedelta
import datetime
import json  # added for JSON output
import time
import numpy as np

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
gen = Generator(lines_info, cp_depot_distances, depots)
initial_solution, _ = gen.generate_initial_set()

# Connection Network
graph_builder = GraphBuilder('initializer/files/timetables.csv')
graph = graph_builder.build_graph()

def generate_columns(S, graph, depots, dh_df, dh_times_df, Z_min, K, I):
    
    cnt = 0
    Z_values = []
    lens_of_s = []
    
    while cnt <= I:
        status, model, current_cost, duals = run_rmp(
            depots,
            S,
            timetables_csv="initializer/files/timetables.csv",
            cplex_path="/Applications/CPLEX_Studio2211/cplex/bin/arm64_osx/cplex",
            log_filename="rmp_output.log"
        )
        
        Z_values.append(current_cost)
        
        optimal = min(Z_values)
        
        if abs(current_cost - optimal) < Z_min:
            cnt += 1

        print("Done. Results also in rmp_output.log")

        # Reduced costs calculations
        graph = add_reduced_cost_info(graph, duals, dh_times_df)
        print(graph)

        # SPFA Run
        trip_keys = list(map(lambda x: int(x.split("_")[-1]), list(S.keys())))
           
        last_route_number = max(trip_keys)
        source_nodes = [n for n, d in graph.nodes(data=True) if d.get("type")=="K"]
        all_labels = {}
        times = []
        for i, t in enumerate(source_nodes):
            print(f"Running SPFA for source {t} ({i+1} of {len(source_nodes)})...")
            start_time = time.time()
            all_labels[t] = run_spfa(graph, t, D=120, T_d=19*60, dh_df=dh_df, last_route_number=last_route_number)
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
        top10 = flat[:K]

        # Build new entries, renaming keys to continue from last_route_number+1
        new_entries = {}
        for i, (old_key, info) in enumerate(top10, start=last_route_number+1):
            if info["ReducedCost"] >= 0:
                continue
            new_key = f"Route_{i}"
            new_entries[new_key] = {
                "Path": info["Path"],
                "Cost": info["Cost"],
                "Data": info["Data"]
            }
            print(f"{new_key} (was {old_key}): ReducedCost = {info['ReducedCost']:.2f}, Path = {' → '.join(info['Path'])}")

        # Merge into S ---
        S.update(new_entries)
        lens_of_s.append(len(S))

    
    print("NEW ENTRIES -------------------------")
    print(new_entries)
    return S, optimal, Z_values, lens_of_s

columns, optimal, Z_values, lens_of_s = generate_columns(
    S=initial_solution,
    graph=graph,
    depots=depots,
    dh_df=dh_df,
    dh_times_df=dh_times_df,
    Z_min=50,
    K=10,
    I=2
)

print(Z_values)
print(lens_of_s)

final_solution, optimal_fitness = run_ga(
    S=initial_solution,
    Lmin=140,
    Lmax=180,
    pop_size=100,
    max_gen=20000,
    crossover_prob=0.3,
    mutation_prob=0.01,
    elite_size=5
)

print(optimal_fitness)



