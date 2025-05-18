import sys
import pulp as pl
import pandas as pd
from initializer.generator import Generator
from initializer.utils import group_routes_by_depot

# === Logger class, unchanged ===
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

# === Build function ===
def build_rmp_model(
    lines_info: dict,
    cp_depot_distances: dict,
    depots: dict,
    timetables_csv: str = "initializer/files/timetables.csv",
    name: str = "RestrictedMasterProblem"
):
    """
    Build the RMP model and return:
      model, X_vars, cost_map, coverage_map, trips_array, grouped_routes
    """
    gen = Generator(lines_info, cp_depot_distances, depots)
    initial_solution, _ = gen.generate_initial_set()
    grouped_routes, routes_costs = group_routes_by_depot(initial_solution)

    trips = pd.read_csv(timetables_csv, usecols=["trip_id"])["trip_id"].values

    model = pl.LpProblem(name, pl.LpMinimize)

    # decision vars & costs
    X = {}
    C = {}
    for k, depot in enumerate(grouped_routes):
        for p, cost in enumerate(routes_costs[depot]):
            X[(k,p)] = pl.LpVariable(f"{depot}_col{p}", 0, 1, cat="Continuous")
            C[(k,p)] = cost

    # coverage matrix
    coverage = {}
    for k, (depot, routes) in enumerate(grouped_routes.items()):
        for p, route in enumerate(routes):
            for i, trip in enumerate(trips):
                coverage[(k,p,i)] = int(trip in route)

    # α constraints
    for i, trip in enumerate(trips):
        model += (
            pl.lpSum(coverage[(k,p,i)] * X[(k,p)]
                     for k in range(len(grouped_routes))
                     for p in range(len(grouped_routes[list(grouped_routes.keys())[k]])))
            == 1,
            f"alpha_trip_{trip}"
        )

    # β constraints
    for k, depot in enumerate(grouped_routes):
        model += (
            pl.lpSum(X[(k,p)] for p in range(len(grouped_routes[depot])))
            <= depots[depot]["capacity"],
            f"beta_depot_{depot}"
        )

    # objective
    model += pl.lpSum(C[key] * X[key] for key in X)

    return model, X, C, coverage, trips, grouped_routes

# === Solve function ===
def solve_rmp_model(
    model: pl.LpProblem,
    cplex_path: str = "/Applications/CPLEX_Studio2211/cplex/bin/arm64_osx/cplex"
):
    """
    Solve the model, extract status, objective, and duals dict.
    """
    solver = pl.CPLEX_CMD(path=cplex_path)
    status = model.solve(solver)

    obj = pl.value(model.objective)

    alpha = {}
    beta = {}
    for name, constr in model.constraints.items():
        if name.startswith("alpha_trip_"):
            trip = name.replace("alpha_trip_","")
            alpha[trip] = constr.pi
        elif name.startswith("beta_depot_"):
            depot = name.replace("beta_depot_","")
            beta[depot] = constr.pi

    return status, obj, {"alpha": alpha, "beta": beta}

# === Wrapper that keeps your prints + logger ===
def run_rmp(
    lines_info: dict,
    cp_depot_distances: dict,
    depots: dict,
    timetables_csv: str = "initializer/files/timetables.csv",
    cplex_path: str = "/Applications/CPLEX_Studio2211/cplex/bin/arm64_osx/cplex",
    log_filename: str = "rmp_output.log"
):
    """
    Redirects stdout to Logger, prints sections, solves RMP, and returns
    (model, objective, duals).
    """
    # install logger
    logger = Logger(log_filename)
    original_stdout = sys.stdout
    sys.stdout = logger

    try:
        print("="*85)
        print("SECTION 1: INITIAL SOLUTION GENERATION PROCESS")
        print("="*85)

        model, X, C, coverage, trips, grouped = build_rmp_model(
            lines_info, cp_depot_distances, depots, timetables_csv
        )

        print()
        print("="*85)
        print("SECTION 2: RMP SOLVING RESULTS")
        print("="*85)

        status, obj, duals = solve_rmp_model(model, cplex_path)

        # print X values
        print("\n--- Routes Coefficients ---")
        for (k,p), var in X.items():
            print(f"X[{k},{p}] = {var.varValue:.4f} (cost = {C[(k,p)]:.4f})")

        # coverage check
        print("\n--- Trip Coverage Check ---")
        for i, trip in enumerate(trips):
            lhs = sum(coverage[(k,p,i)] * X[(k,p)].varValue
                      for k in range(len(grouped))
                      for p in range(len(grouped[list(grouped.keys())[k]])))
            print(f"Trip {trip}: coverage = {lhs:.4f}")

        print(f"\nZ (Objective Value): {obj:.4f}")

        # duals
        print("\n--- Dual Values ---")
        for trip, π in duals["alpha"].items():
            print(f"α for trip {trip}: {π:.4f}")
        for depot, π in duals["beta"].items():
            print(f"β for depot {depot}: {π:.4f}")

    finally:
        # restore stdout and close log
        sys.stdout = original_stdout
        logger.log.close()

    return status, model, obj, duals

from initializer.inputs import lines_info, cp_depot_distances, depots

status, model, optimal_cost, duals = run_rmp(
    lines_info,
    cp_depot_distances,
    depots,
    timetables_csv="initializer/files/timetables.csv",
    cplex_path="/Applications/CPLEX_Studio2211/cplex/bin/arm64_osx/cplex",
    log_filename="rmp_output.log"
)

print("Done—results also in rmp_output.log")


from initializer.inputs import *
from initializer.utils import *

lines_info = lines_info
cp_depot_distances = cp_depot_distances
cp_locations_summary = summarize_cp_locations(lines_info)
cp_distances = calculate_cp_distances(cp_locations_summary, lines_info)
transformed_cp_depot_distances = transform_cp_depot_distances(cp_depot_distances)
dh_dict = merge_distances_dicts(transformed_cp_depot_distances, cp_distances)
dh_df = make_deadhead_df(dh_dict)
dh_times_df = make_deadhead_times_df(20, dh_df)


from initializer.conn_network_builder import GraphBuilder
builder = GraphBuilder('initializer/files/timetables.csv')
graph = builder.build_graph()

for u, v, data in graph.edges(data=True):
    if graph.nodes[v]["type"] == "T":
        dh_cost = data["time"] * 1.6
        reduced_cost = dh_cost - duals["alpha"][graph.nodes[v]['id']]
    elif graph.nodes[v]["type"] == "K":
        dh_cost = data["time"] * 1.6
        reduced_cost = dh_cost + duals["beta"][v]
    else:
        j = v.replace("c", "d")
        end_cp = graph.nodes[u]["end_cp"]
        dh_time = float(dh_times_df.loc[end_cp, j])
        reduced_cost = dh_time * 1.6
    graph.edges[u, v]["reduced_cost"] = reduced_cost
    # ISSO ACIMA 'E A MATRIZ DE ADJACENCIA (gerada como um grafo mesmo) QUE PRECISA AGORA SER PERCORRIDA PELO SPFA.


# from collections import deque
# from datetime import timedelta

# def run_spfa(graph, source_node, D, T_d):
#     # Initialize reduced costs
#     red_costs = {n: math.inf for n in graph.nodes}
#     red_costs[source_node] = 0

#     # Initialize distance, time, and “since charge” trackers
#     u_planned = graph.nodes[source_node]["planned_travel_time"]
#     u_start_cp = graph.nodes[source_node]['start_cp']
#     u_end_cp = graph.nodes[source_node]['end_cp']
#     u_travel_dist = dh_df.loc[u_start_cp, u_end_cp]

#     dist = {n: None for n in graph.nodes}
#     dist[source_node] = u_travel_dist

#     time = {n: None for n in graph.nodes}
#     time[source_node] = u_planned

#     current_time = {n: None for n in graph.nodes}
#     current_time[source_node] = graph.nodes[source_node]["end_time"]

#     dist_since_charge = {n: None for n in graph.nodes}
#     dist_since_charge[source_node] = u_travel_dist

#     queue = deque([source_node])
#     iteration = 1

#     while queue:
#         u = queue.popleft()
#         print(f"Iteration {iteration}, queue size {len(queue)}, current node: {u}")
#         for v in graph.neighbors(u):
#             arc_cost = graph[u][v]["reduced_cost"]
#             arc_time = graph[u][v]["time"]
#             arc_dist = graph[u][v]["dist"]

#             # If u is a charging‐station node, only allow arcs to trips that start after arrival
#             if u.startswith("c"):
#                 if graph.nodes[v]['start_time'] < current_time[u] + timedelta(minutes=arc_time):
#                     continue
#             # Otherwise, only consider arcs to “trip” nodes whose names start with 'l'
#             elif not v.startswith("l"):
#                 continue

#             # Gather v’s metadata
#             v_start = graph.nodes[v]["start_time"]
#             v_planned = graph.nodes[v]["planned_travel_time"]
#             v_start_cp = graph.nodes[v]['start_cp']
#             v_end_cp = graph.nodes[v]['end_cp']
#             v_dist = dh_df.loc[v_start_cp, v_end_cp]
#             v_end_time = graph.nodes[v]["end_time"]

#             # Compute waiting time (in minutes)
#             wait = (v_start - (current_time[u] + timedelta(minutes=arc_time))).total_seconds() / 60

#             # 1) Time‐limit check
#             if time[u] + arc_time + wait + v_planned >= T_d:
#                 continue

#             # 2) Distance‐since‐last‐charge check
#             if dist_since_charge[u] + arc_dist + v_dist > D:
#                 # Need to insert a charging‐station node
#                 depot_cols = dh_df.filter(regex=r"^d\d+_\d+$").columns
#                 best = dh_df.loc[graph.nodes[u]['end_cp'], depot_cols].idxmin()
#                 cs = "c" + best[1:]  # e.g. 'd12_5' → 'c12_5'

#                 # Metrics for arc u→cs
#                 cost_cs = graph[u][cs]["reduced_cost"]
#                 dist_cs = graph[u][cs]["dist"]
#                 time_cs = graph[u][cs]["time"]
#                 # charging_time formula as before
#                 time_since_charge = ((dist_since_charge[u] + dist_cs) / 20) * 60
#                 charge_dur = 15.6 * time_since_charge / 30
#                 total_time_cs = time_cs + charge_dur

#                 # Relax u→cs if improved
#                 new_cost = red_costs[u] + cost_cs
#                 if new_cost < red_costs.get(cs, math.inf):
#                     red_costs[cs] = new_cost
#                     dist[cs] = dist[u] + dist_cs
#                     time[cs] = time[u] + total_time_cs
#                     current_time[cs] = current_time[u] + timedelta(minutes=total_time_cs)
#                     dist_since_charge[cs] = 0
#                     if cs not in queue:
#                         queue.append(cs)
#                 continue

#             # 3) Normal relaxation to v
#             new_cost_v = red_costs[u] + arc_cost
#             if new_cost_v < red_costs[v]:
#                 red_costs[v] = new_cost_v
#                 dist[v] = dist[u] + arc_dist + v_dist
#                 time[v] = time[u] + arc_time + wait + v_planned
#                 current_time[v] = v_end_time
#                 dist_since_charge[v] = dist_since_charge[u] + arc_dist + v_dist
#                 if v not in queue:
#                     queue.append(v)

#         iteration += 1

#     return red_costs

# # Run SPFA from each trip node:
# trip_nodes = [n for n, attr in graph.nodes(data=True) if attr.get("type") == "T"]
# for src in trip_nodes:
#     costs = run_spfa(graph, src, D=120, T_d=19*60)
#     print(f"Source {src} → reduced costs: {costs}")





# para cada K:
    # para cada nó i \in T:
        # SPFA, ensuring dist and time. 
            # OBS: na regra de dominancia, o nó que cobre maior distância e tem menor custo, prevalece.
            # OBS2: como ele nao recebe dest_node, ele acaba achando sp para todos os nós.
            # OBS3: restricao de capacidade de depositos ja é resolvida no RMP.


# import math
# from collections import deque, namedtuple
# from datetime import timedelta

# # a single “state” (label) in the network
# Label = namedtuple("Label", [
#     "node",             # current node ID
#     "cost",             # reduced cost so far (c_p)
#     "dist_since_charge",# distance since last charge/depot (d_p)
#     "time",             # cumulative travel time since depot
#     "current_time",     # actual clock time at arrival
#     "path"              # list of nodes visited so far
# ])

# def run_label_correcting(graph, source, D, T_d):
#     # initialize label‐lists
#     labels = {n: [] for n in graph.nodes}
    
#     # initial label from source depot/trip
#     ss, es = graph.nodes[source]["start_cp"], graph.nodes[source]["end_cp"]
#     init_dist = dh_df.loc[ss, es]
#     init_time = graph.nodes[source]["planned_travel_time"]
#     init_clock = graph.nodes[source]["end_time"]
#     init = Label(
#         node=source,
#         cost=0.0,
#         dist_since_charge=init_dist,
#         time=init_time,
#         current_time=init_clock,
#         path=[source]
#     )
#     labels[source].append(init)
    
#     # FIFO queue of labels to process
#     queue = deque([init])
    
#     while queue:
#         lbl = queue.popleft()
#         u = lbl.node
        
#         for v in graph.neighbors(u):
#             # skip invalid arcs (only Cs→trips if timing ok, else only trip→trip)
#             arc_time = graph[u][v]["time"]
#             arc_dist = graph[u][v]["dist"]
#             if u.startswith("c"):
#                 if graph.nodes[v]["start_time"] < lbl.current_time + timedelta(minutes=arc_time):
#                     continue
#             elif not v.startswith("l"):
#                 continue
            
#             # compute wait & next‐trip stats
#             wait = (graph.nodes[v]["start_time"] - 
#                     (lbl.current_time + timedelta(minutes=arc_time))).total_seconds() / 60
#             v_planned = graph.nodes[v]["planned_travel_time"]
#             sc, ec = graph.nodes[v]["start_cp"], graph.nodes[v]["end_cp"]
#             v_dist = dh_df.loc[sc, ec]
#             v_end_clock = graph.nodes[v]["end_time"]
            
#             # total time & distance if we go directly u→v
#             tot_time = lbl.time + arc_time + wait + v_planned
#             tot_dist = lbl.dist_since_charge + arc_dist + v_dist
            
#             # 1) time‐window constraint
#             if tot_time > T_d:
#                 continue
            
#             # 2) distance constraint → need a charge node first
#             if tot_dist > D:
#                 # pick best depot/CS to recharge
#                 depot_cols = dh_df.filter(regex=r"^d\d+_\d+$").columns
#                 best = dh_df.loc[graph.nodes[u]["end_cp"], depot_cols].idxmin()
#                 cs = "c" + best[1:]  # e.g. 'd12_5' → 'c12_5'
                
#                 # metrics for u→cs
#                 cost_cs = graph[u][cs]["reduced_cost"]
#                 dist_cs = graph[u][cs]["dist"]
#                 time_cs = graph[u][cs]["time"]
#                 # charging duration
#                 charge_minutes = 15.6 * (((lbl.dist_since_charge + dist_cs) / 20) * 60) / 30
#                 new_clock = lbl.current_time + timedelta(minutes=time_cs + charge_minutes)
                
#                 new_lbl = Label(
#                     node=cs,
#                     cost=lbl.cost + cost_cs,
#                     dist_since_charge=0.0,
#                     time=lbl.time + time_cs + charge_minutes,
#                     current_time=new_clock,
#                     path=lbl.path + [cs]
#                 )
#                 dest = cs
#             else:
#                 # normal trip extension u→v
#                 cost_uv = graph[u][v]["reduced_cost"]
#                 new_lbl = Label(
#                     node=v,
#                     cost=lbl.cost + cost_uv,
#                     dist_since_charge=tot_dist,
#                     time=tot_time,
#                     current_time=v_end_clock,
#                     path=lbl.path + [v]
#                 )
#                 dest = v
            
#             # Dominance test: discard if dominated by any existing label at dest
#             dominated = False
#             to_remove = []
#             for existing in labels[dest]:
#                 if (existing.cost <= new_lbl.cost and
#                     existing.dist_since_charge <= new_lbl.dist_since_charge and
#                     existing.current_time <= new_lbl.current_time):
#                     # existing label is better or equal in all resources
#                     dominated = True
#                     break
#                 if (new_lbl.cost <= existing.cost and
#                     new_lbl.dist_since_charge <= existing.dist_since_charge and
#                     new_lbl.current_time <= existing.current_time):
#                     # new label dominates existing
#                     to_remove.append(existing)
            
#             if dominated:
#                 continue
            
#             # remove any labels dominated by new_lbl
#             for ex in to_remove:
#                 labels[dest].remove(ex)
            
#             # keep new_lbl and enqueue it
#             labels[dest].append(new_lbl)
#             queue.append(new_lbl)
    
#     return labels

# import time
# import numpy as np
# # example usage: generate labels (columns) from each trip-node
# trip_nodes = [n for n, d in graph.nodes(data=True) if d.get("type")=="K"]
# all_labels = {}
# times = []
# for i, t in enumerate(trip_nodes):
#     print(f"Running SPFA for trip {t} ({i+1} of {len(trip_nodes)})...")
#     start_time = time.time()
#     all_labels[t] = run_label_correcting(graph, t, D=120, T_d=19*60)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     times.append(elapsed_time)
#     print(f"SPFA for trip {t} is Done!")
#     print(f"Running time: {times[-1]}")
#     print(f"Remaining time prediction: {np.mean(times) * len(trip_nodes) - (i+1)}.")
#     print(f"Total time prediction: {np.mean(times) * len(trip_nodes)}.")
#     print()
# now all_labels[t] holds the nondominated labels (routes) from t :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}:contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}


# import math
# from collections import deque, namedtuple
# from datetime import timedelta

# # a single “state” (label) in the network
# Label = namedtuple("Label", [
#     "node",             # current node ID
#     "cost",             # reduced cost so far (c_p)
#     "dist_since_charge",# distance since last charge/depot (d_p)
#     "time",             # cumulative travel time since depot
#     "current_time",     # actual clock time at arrival
#     "path"              # list of nodes visited so far
# ])

# def run_label_correcting(graph, source, D, T_d):
#     # initialize label‐lists
#     labels = {n: [] for n in graph.nodes}
    
#     # initial label from source depot/trip
#     #ss, es = graph.nodes[source]["start_cp"], graph.nodes[source]["end_cp"]
#     #init_dist = dh_df.loc[ss, es]
#     #init_time = graph.nodes[source]["planned_travel_time"]
#     #init_clock = graph.nodes[source]["end_time"]
#     init = Label(
#         node=source,
#         cost=0.0,
#         dist_since_charge=0,
#         time=0,
#         current_time=None,
#         path=[source]
#     )
#     labels[source].append(init)
    
#     # FIFO queue of labels to process
#     queue = deque([init])
    
#     while queue:
#         lbl = queue.popleft()
#         u = lbl.node
        
#         for v in graph.neighbors(u):
#             # skip invalid arcs (only Cs→trips if timing ok, else only trip→trip)
#             arc_time = graph[u][v]["time"]
#             arc_dist = graph[u][v]["dist"]

#             if lbl.current_time is None:
#                 clock = graph.nodes[v]["end_time"]
#             else:
#                 clock = lbl.current_time

#             if u.startswith("c"):
#                 if graph.nodes[v]["start_time"] < clock + timedelta(minutes=arc_time):
#                     continue
#             elif not v.startswith("l"):
#                 continue
            
#             # compute wait & next‐trip stats
#             wait = (graph.nodes[v]["start_time"] - 
#                     (clock + timedelta(minutes=arc_time))).total_seconds() / 60
#             v_planned = graph.nodes[v]["planned_travel_time"]
#             sc, ec = graph.nodes[v]["start_cp"], graph.nodes[v]["end_cp"]
#             v_dist = dh_df.loc[sc, ec]
#             v_end_clock = graph.nodes[v]["end_time"]
            
#             # total time & distance if we go directly u→v
#             tot_time = lbl.time + arc_time + wait + v_planned
#             tot_dist = lbl.dist_since_charge + arc_dist + v_dist
            
#             # 1) time‐window constraint
#             if tot_time > T_d:
#                 continue
            
#             # 2) distance constraint → need a charge node first
#             if tot_dist > D:
#                 # pick best depot/CS to recharge
#                 depot_cols = dh_df.filter(regex=r"^d\d+_\d+$").columns
#                 best = dh_df.loc[graph.nodes[u]["end_cp"], depot_cols].idxmin()
#                 cs = "c" + best[1:]  # e.g. 'd12_5' → 'c12_5'
                
#                 # metrics for u→cs
#                 cost_cs = graph[u][cs]["reduced_cost"]
#                 dist_cs = graph[u][cs]["dist"]
#                 time_cs = graph[u][cs]["time"]
#                 # charging duration
#                 charge_minutes = 15.6 * (((lbl.dist_since_charge + dist_cs) / 20) * 60) / 30
#                 new_clock = clock + timedelta(minutes=time_cs + charge_minutes)
                
#                 new_lbl = Label(
#                     node=cs,
#                     cost=lbl.cost + cost_cs,
#                     dist_since_charge=0.0,
#                     time=lbl.time + time_cs + charge_minutes,
#                     current_time=new_clock,
#                     path=lbl.path + [cs]
#                 )
#                 dest = cs
#             else:
#                 # normal trip extension u→v
#                 cost_uv = graph[u][v]["reduced_cost"]
#                 new_lbl = Label(
#                     node=v,
#                     cost=lbl.cost + cost_uv,
#                     dist_since_charge=tot_dist,
#                     time=tot_time,
#                     current_time=v_end_clock,
#                     path=lbl.path + [v]
#                 )
#                 dest = v
            
#             # Dominance test: discard if dominated by any existing label at dest
#             dominated = False
#             to_remove = []
#             for existing in labels[dest]:
#                 if (existing.cost <= new_lbl.cost and
#                     existing.dist_since_charge <= new_lbl.dist_since_charge and
#                     existing.current_time <= new_lbl.current_time):
#                     # existing label is better or equal in all resources
#                     dominated = True
#                     break
#                 if (new_lbl.cost <= existing.cost and
#                     new_lbl.dist_since_charge <= existing.dist_since_charge and
#                     new_lbl.current_time <= existing.current_time):
#                     # new label dominates existing
#                     to_remove.append(existing)
            
#             if dominated:
#                 continue
            
#             # remove any labels dominated by new_lbl
#             for ex in to_remove:
#                 labels[dest].remove(ex)
            
#             # keep new_lbl and enqueue it
#             labels[dest].append(new_lbl)
#             queue.append(new_lbl)
    
#     return labels

# import time
# import numpy as np
# # example usage: generate labels (columns) from each trip-node
# source_nodes = [n for n, d in graph.nodes(data=True) if d.get("type")=="K"]
# all_labels = {}
# times = []
# for i, t in enumerate(source_nodes):
#     print(f"Running SPFA for source {t} ({i+1} of {len(source_nodes)})...")
#     start_time = time.time()
#     all_labels[t] = run_label_correcting(graph, t, D=120, T_d=19*60)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     times.append(elapsed_time)
#     print(f"SPFA for source {t} successfuly finished! Duration: {elapsed_time}.")
#     print(f"Running time: {times[-1]}")
#     print(f"Remaining time prediction: {np.mean(times) * (len(source_nodes) - (i+1))}.")
#     print(f"Total time prediction: {np.mean(times) * len(source_nodes)}.")
#     print()
# print(f"SPFA was done in {np.sum(times)} seconds.")
# print()
# print(all_labels)
# # now all_labels[t] holds the nondominated labels (routes) from t :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}:contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}

import math
from collections import deque, namedtuple
from datetime import timedelta

# one label = one partial route
Label = namedtuple("Label", [
    "node",             # current node ID
    "cost",             # accumulated reduced cost
    "dist_since_charge",# km since last recharge
    "time",             # minutes since departure from depot
    "current_time",     # actual datetime at arrival
    "path"              # list of node IDs visited
])

def run_label_correcting(graph, D, T_d, K=None):
    """
    Multi‐label SPFA seeded via first‐leg from all depots.

    Args:
      graph: NetworkX DiGraph with node types "K","T","C"
      D: max km between charges
      T_d: max minutes for a route
      K: if set, stop after K complete depot→depot routes
    Returns:
      results: list of completed Label objects (returning to a depot)
    """
    # prepare
    depot_nodes = [n for n,d in graph.nodes(data=True) if d["type"]=="K"]
    labels_at = {n: [] for n in graph.nodes}   # nondominated labels per node
    queue = deque()
    results = []

    # === Option A seeding: pre‐relax depot→trip arcs ===
    for d in depot_nodes:
        for v in graph.neighbors(d):
            if graph.nodes[v]["type"]!="T":
                continue
            e = graph.edges[d, v]
            arc_cost = e["reduced_cost"]
            arc_time = e["time"]
            arc_dist = e["dist"]

            vdata = graph.nodes[v]
            v_cp_start = vdata["start_cp"]
            v_cp_end   = vdata["end_cp"]
            v_dist     = dh_df.loc[v_cp_start, v_cp_end]
            v_planned  = vdata["planned_travel_time"]
            v_end_time= vdata["end_time"]

            tot_time = arc_time + v_planned
            tot_dist = arc_dist + v_dist
            clock    = v_end_time

            lbl = Label(
                node=v,
                cost=arc_cost,
                dist_since_charge=tot_dist,
                time=tot_time,
                current_time=clock,
                path=[d, v]
            )
            labels_at[v].append(lbl)
            queue.append(lbl)

    # === Main label‐correcting loop ===
    while queue:
        lbl = queue.popleft()
        u = lbl.node

        # If we've returned to a depot, record it
        if u in depot_nodes and len(lbl.path)>1:
            results.append(lbl)
            if K and len(results)>=K:
                break
            # continue exploring in case we want more columns

        for v in graph.neighbors(u):
            vnode = graph.nodes[v]
            etype = vnode["type"]

            # allow C→T only if timing ok
            if graph.nodes[u]["type"]=="C":
                arc_time = graph.edges[u,v]["time"]
                if vnode["start_time"] < lbl.current_time + timedelta(minutes=arc_time):
                    continue
            # otherwise only T→T or T→C or C→C
            elif etype!="T" and etype!="C":
                continue

            e = graph.edges[u,v]
            arc_cost = e["reduced_cost"]
            arc_time = e["time"]
            arc_dist = e["dist"]

            # compute next‐trip stats
            if etype=="T":
                wait = (vnode["start_time"] - 
                        (lbl.current_time + timedelta(minutes=arc_time))).total_seconds()/60
                wait = max(0, wait)
                v_planned = vnode["planned_travel_time"]
                sc, ec = vnode["start_cp"], vnode["end_cp"]
                v_dist = dh_df.loc[sc, ec]
                v_end_time = vnode["end_time"]

                new_time = lbl.time + arc_time + wait + v_planned
                new_dist = lbl.dist_since_charge + arc_dist + v_dist
                new_clock= v_end_time
                new_cost = lbl.cost + arc_cost
                dest = v

            else:  # etype=="C": insert charging node
                # pick best depot to recharge
                depot_cols = dh_df.filter(regex=r"^d\d+_\d+$").columns
                best = dh_df.loc[ graph.nodes[u]["end_cp"], depot_cols ].idxmin()
                cs   = "c" + best[1:]
                e_cs = graph.edges[u,cs]
                arc_cost_cs = e_cs["reduced_cost"]
                arc_time_cs = e_cs["time"]
                arc_dist_cs = e_cs["dist"]

                # charging duration
                charged_minutes = 15.6 * (((lbl.dist_since_charge + arc_dist_cs)/20)*60)/30
                new_clock = lbl.current_time + timedelta(minutes=arc_time_cs + charged_minutes)

                new_time = lbl.time + arc_time_cs + charged_minutes
                new_dist = 0.0
                new_cost = lbl.cost + arc_cost_cs
                dest = cs

            # enforce resource constraints
            if new_time > T_d:
                continue

            # Note: we already handled distance > D via the charging branch above

            # create and dominate‐prune the new label
            new_lbl = Label(
                node=dest,
                cost=new_cost,
                dist_since_charge=new_dist,
                time=new_time,
                current_time=new_clock,
                path=lbl.path + [dest]
            )

            # dominance check
            dominated = False
            to_remove = []
            for ex in labels_at[dest]:
                if (ex.cost <= new_lbl.cost and
                    ex.dist_since_charge <= new_lbl.dist_since_charge and
                    ex.current_time <= new_lbl.current_time):
                    dominated = True
                    break
                if (new_lbl.cost <= ex.cost and
                    new_lbl.dist_since_charge <= ex.dist_since_charge and
                    new_lbl.current_time <= ex.current_time):
                    to_remove.append(ex)
            if dominated:
                continue
            for ex in to_remove:
                labels_at[dest].remove(ex)

            # keep & enqueue
            labels_at[dest].append(new_lbl)
            queue.append(new_lbl)

    return results

import time
import numpy as np
# example usage: generate labels (columns) from each trip-node
source_nodes = [n for n, d in graph.nodes(data=True) if d.get("type")=="K"]
all_labels = {}
times = []

start_time = time.time()
RESULTS = run_label_correcting(graph, D=120, T_d=19*60)
end_time = time.time()
elapsed_time = end_time - start_time
times.append(elapsed_time)

print(f"SPFA was done in {np.sum(times)} seconds.")
print()
print(RESULTS)
