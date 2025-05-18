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
class ColumnGenerator:

    def __init__(self, lines_info, cp_depot_distances, depots):
        self.lines_info = lines_info
        self.cp_depot_distances = cp_depot_distances
        self.cp_depots = depots
        self.timetables_csv: str = "initializer/files/timetables.csv",
        self.name: str = "RestrictedMasterProblem"
    
    def build_model()
    gen = Generator(self.lines_info, self.cp_depot_distances, self.depots)
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

print("Done. Results also in rmp_output.log")


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

import math
from collections import deque, namedtuple
from datetime import timedelta
import datetime
import json  # added for JSON output

# a single “state” (label) in the network
Label = namedtuple("Label", [
    "node",             # current node ID
    "cost",             # reduced cost so far (c_p)
    "dist_since_charge",# distance since last charge/depot (d_p)
    "time",             # cumulative travel time since depot
    "current_time",     # actual clock time at arrival
    "path"              # list of nodes visited so far
])

def run_label_correcting(graph, source, D, T_d):
    # initialize label‐lists
    labels = {n: [] for n in graph.nodes}
    
    # initial label from source depot
    init = Label(
        node=source,
        cost=0.0,
        dist_since_charge=0,
        time=0,
        current_time=None,
        path=[source]
    )
    labels[source].append(init)
    
    # FIFO queue of labels to process
    queue = deque([init])
    
    while queue:
        lbl = queue.popleft()
        u = lbl.node
        
        for v in graph.neighbors(u):
            arc_time = graph[u][v]["time"]
            arc_dist = graph[u][v]["dist"]

            if lbl.current_time is None:
                # first hop: back out the travel time so wait = 0
                clock = graph.nodes[v]["start_time"] - timedelta(minutes=arc_time)
            else:
                clock = lbl.current_time

            if u.startswith("c"):
                if graph.nodes[v]["start_time"] < clock + timedelta(minutes=arc_time):
                    continue
            elif not v.startswith("l"):
                continue
            
            # compute wait & next‐trip stats
            wait = (graph.nodes[v]["start_time"] - 
                    (clock + timedelta(minutes=arc_time))).total_seconds() / 60
            v_planned = graph.nodes[v]["planned_travel_time"]
            sc, ec = graph.nodes[v]["start_cp"], graph.nodes[v]["end_cp"]
            v_dist = dh_df.loc[sc, ec]
            v_end_clock = graph.nodes[v]["end_time"]
            
            tot_time = lbl.time + arc_time + wait + v_planned
            tot_dist = lbl.dist_since_charge + arc_dist + v_dist
            
            # 1) time‐window constraint
            if tot_time > T_d:
                continue
            
            # 2) distance‐since‐charge constraint
            if tot_dist > D:
                depot_cols = dh_df.filter(regex=r"^d\d+_\d+$").columns
                best = dh_df.loc[graph.nodes[u]["end_cp"], depot_cols].idxmin()
                cs = "c" + best[1:]
                
                cost_cs = graph[u][cs]["reduced_cost"]
                dist_cs = graph[u][cs]["dist"]
                time_cs = graph[u][cs]["time"]
                charge_minutes = 15.6 * (((lbl.dist_since_charge + dist_cs) / 20) * 60) / 30
                new_clock = clock + timedelta(minutes=time_cs + charge_minutes)
                
                new_lbl = Label(
                    node=cs,
                    cost=lbl.cost + cost_cs,
                    dist_since_charge=0.0,
                    time=lbl.time + time_cs + charge_minutes,
                    current_time=new_clock,
                    path=lbl.path + [cs]
                )
                dest = cs
            else:
                cost_uv = graph[u][v]["reduced_cost"]
                new_lbl = Label(
                    node=v,
                    cost=lbl.cost + cost_uv,
                    dist_since_charge=tot_dist,
                    time=tot_time,
                    current_time=v_end_clock,
                    path=lbl.path + [v]
                )
                dest = v
            
            # Dominance test
            dominated = False
            to_remove = []
            for existing in labels[dest]:
                fixed_ct = existing.current_time or datetime.datetime.min
                if (existing.cost <= new_lbl.cost and
                    existing.dist_since_charge <= new_lbl.dist_since_charge and
                    fixed_ct <= new_lbl.current_time):
                    dominated = True
                    break
                if (new_lbl.cost <= existing.cost and
                    new_lbl.dist_since_charge <= existing.dist_since_charge and
                    new_lbl.current_time <= fixed_ct):
                    to_remove.append(existing)
            if dominated:
                continue
            for ex in to_remove:
                labels[dest].remove(ex)
            
            labels[dest].append(new_lbl)
            queue.append(new_lbl)
    
    # Option B: close every trip‐ending label back to this source depot
    completed_routes = []
    for lbl_list in labels.values():
        for lbl in lbl_list:
            if lbl.node.startswith("l"):
                # add the reduced cost of the final arc i->d
                ret_cost = graph[lbl.node][source]["reduced_cost"]
                full_cost = lbl.cost + ret_cost
                full_path = lbl.path + [source]
                completed_routes.append(
                    Label(
                        node=source,
                        cost=full_cost,
                        dist_since_charge=lbl.dist_since_charge,
                        time=lbl.time,
                        current_time=lbl.current_time,
                        path=full_path
                    )
                )
    return completed_routes

# example usage remains unchanged
import time
import numpy as np

source_nodes = [n for n, d in graph.nodes(data=True) if d.get("type")=="K"]
all_labels = {}
times = []
for i, t in enumerate(source_nodes):
    print(f"Running SPFA for source {t} ({i+1} of {len(source_nodes)})...")
    start_time = time.time()
    all_labels[t] = run_label_correcting(graph, t, D=120, T_d=19*60)
    end_time = time.time()
    elapsed_time = end_time - start_time
    times.append(elapsed_time)
    print(f"SPFA for source {t} successfully finished! Duration: {elapsed_time:.2f}s.")
    print(f"Remaining time prediction: {np.mean(times) * (len(source_nodes) - (i+1)):.2f}s.")
    print(f"Total time prediction: {np.mean(times) * len(source_nodes):.2f}s.\n")

print(f"SPFA was done in {np.sum(times):.2f}s.\n")

# ---- serialize and save to JSON ----
serializable = {}
for source, routes in all_labels.items():
    serializable[source] = []
    for lbl in routes:
        serializable[source].append({
            "node": lbl.node,
            "cost": lbl.cost,
            "dist_since_charge": lbl.dist_since_charge,
            "time": lbl.time,
            "current_time": lbl.current_time.isoformat() if lbl.current_time else None,
            "path": lbl.path
        })

with open("all_labels.json", "w") as f:
    json.dump(serializable, f, indent=2)

print("Results saved to all_labels.json")
print(all_labels)

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