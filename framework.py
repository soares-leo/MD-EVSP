from initializer.inputs import *
from initializer.generator import Generator
from initializer.utils import *
from initializer.conn_network_builder import GraphBuilder
from column_generator.utils import add_reduced_cost_info
from column_generator.rmp_solver import run_rmp
from column_generator.spfa import run_spfa
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

# RMP Solver
status, model, optimal_cost, duals = run_rmp(
    depots,
    initial_solution,
    timetables_csv="initializer/files/timetables.csv",
    cplex_path="/Applications/CPLEX_Studio2211/cplex/bin/arm64_osx/cplex",
    log_filename="rmp_output.log"
)

print("Done. Results also in rmp_output.log")

# Reduced costs calculations
graph = add_reduced_cost_info(graph, duals, dh_times_df)
print(graph)

# SPFA Run
source_nodes = [n for n, d in graph.nodes(data=True) if d.get("type")=="K"]
all_labels = {}
times = []
for i, t in enumerate(source_nodes):
    print(f"Running SPFA for source {t} ({i+1} of {len(source_nodes)})...")
    start_time = time.time()
    all_labels[t] = run_spfa(graph, t, D=120, T_d=19*60, dh_df=dh_df)
    end_time = time.time()
    elapsed_time = end_time - start_time
    times.append(elapsed_time)
    print(f"SPFA for source {t} successfully finished! Duration: {elapsed_time:.2f}s.")
    print(f"Remaining time prediction: {np.mean(times) * (len(source_nodes) - (i+1)):.2f}s.")
    print(f"Total time prediction: {np.mean(times) * len(source_nodes):.2f}s.\n")

print(f"SPFA was done in {np.sum(times):.2f}s.\n")



# ---- serialize and save to JSON ----
# serializable = {}
# for source, routes in all_labels.items():
#     serializable[source] = []
#     for lbl in routes:
#         serializable[source].append({
#             "node": lbl.node,
#             "cost": lbl.cost,
#             "dist_since_charge": lbl.dist_since_charge,
#             "time": lbl.time,
#             "current_time": lbl.current_time.isoformat() if lbl.current_time else None,
#             "path": lbl.path
#         })

# with open("all_labels.json", "w") as f:
#     json.dump(serializable, f, indent=2)

# print("Results saved to all_labels.json")
# print(all_labels)

self.initial_solution[f"Route_{i}"] = {
    "Path": self.route,
    "Cost": route_cost,
    "Data": {
        "total_dh_dist": self.total_dh_dist,
        "total_dh_time": self.total_dh_time,
        "total_travel_dist": self.total_travel_dist,
        "total_travel_time": self.total_travel_time,
        "total_wait_time": self.total_wait_time,
        "final_time": self.current_time
    }
}