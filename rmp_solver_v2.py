import sys
import math
import pulp as pl
import pandas as pd
import networkx as nx
from collections import deque
from initializer.generator import Generator
from initializer.utils import group_routes_by_depot
from initializer.conn_network_builder import GraphBuilder
from initializer.inputs import lines_info, cp_depot_distances, depots, problem_params

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

# === Label-based SPFA with on-the-fly dominance ===
def spfa_with_dominance(graph: nx.DiGraph, source: str):
    """
    Label-setting SPFA: maintain labels for each node and apply dominance immediately.
    Returns:
      labels_at_node: dict[node, list of labels]
    Each label: dict with keys 'node', 'reduced_cost', 'total_time', 'pred'
    'pred' is predecessor label reference for path reconstruction.
    """
    # initialize labels per node
    labels = {n: [] for n in graph.nodes()}
    # initial label at source
    init_label = {'node': source, 'reduced_cost': 0.0, 'total_time': 0.0, 'pred': None}
    labels[source].append(init_label)

    # queue of labels to extend
    queue = deque([init_label])

    while queue:
        lbl = queue.popleft()
        u = lbl['node']
        # extend along outgoing arcs
        count = 1
        for v, data in graph[u].items():
            print(f"Processing node {count} of {len(graph)}.")
            cost_uv = data.get('reduced_cost', math.inf)
            time_uv = data.get('time', 0)
            new_rc = lbl['reduced_cost'] + cost_uv
            new_time = lbl['total_time'] + time_uv
            # create new candidate label
            new_lbl = {'node': v, 'reduced_cost': new_rc, 'total_time': new_time, 'pred': lbl}
            # check dominance against existing labels at v
            dominated = False
            survivors = []
            for existing in labels[v]:
                # existing dominates new?
                if (existing['reduced_cost'] <= new_lbl['reduced_cost']
                    and existing['total_time'] <= new_lbl['total_time']
                    and (existing['reduced_cost'] < new_lbl['reduced_cost']
                         or existing['total_time'] < new_lbl['total_time'])):
                    dominated = True
                    count+=1
                    break
                # new dominates existing?
                if (new_lbl['reduced_cost'] <= existing['reduced_cost']
                    and new_lbl['total_time'] <= existing['total_time']
                    and (new_lbl['reduced_cost'] < existing['reduced_cost']
                         or new_lbl['total_time'] < existing['total_time'])):
                    # drop existing
                    count+=1
                    continue
                survivors.append(existing)
            if dominated:
                count+=1
                continue
            # keep new label and survivors
            survivors.append(new_lbl)
            labels[v] = survivors
            # only enqueue if node v is to be further expanded (trip or station)
            if graph.nodes[v]['type'] in ('T', 'C'):
                queue.append(new_lbl)
            count+=1
    return labels

# === Recover path from label ===
def reconstruct_path_from_label(lbl):
    path = []
    cur = lbl
    while cur is not None:
        path.append(cur['node'])
        cur = cur['pred']
    return list(reversed(path)), lbl['reduced_cost'], lbl['total_time']

# === Build function (unchanged) ===
def build_rmp_model(
    lines_info: dict,
    cp_depot_distances: dict,
    depots: dict,
    timetables_csv: str = "initializer/files/timetables.csv",
    name: str = "RestrictedMasterProblem"
):
    gen = Generator(lines_info, cp_depot_distances, depots)
    initial_solution, _ = gen.generate_initial_set()
    grouped_routes, routes_costs = group_routes_by_depot(initial_solution)
    trips = pd.read_csv(timetables_csv, usecols=["trip_id"])['trip_id'].values
    model = pl.LpProblem(name, pl.LpMinimize)
    X, C, coverage = {}, {}, {}
    for k, depot in enumerate(grouped_routes):
        for p, cost in enumerate(routes_costs[depot]):
            var = pl.LpVariable(f"{depot}_col{p}", 0, 1)
            X[(k,p)] = var; C[(k,p)] = cost
    for k, (depot, routes) in enumerate(grouped_routes.items()):
        for p, route in enumerate(routes):
            for i, trip in enumerate(trips):
                coverage[(k,p,i)] = int(trip in route)
    for i, trip in enumerate(trips):
        model += (
            pl.lpSum(coverage[(k,p,i)]*X[(k,p)]
                     for k in range(len(grouped_routes))
                     for p in range(len(grouped_routes[ list(grouped_routes.keys())[k] ]))) == 1)
    for k, depot in enumerate(grouped_routes):
        model += ( pl.lpSum(X[(k,p)] for p in range(len(grouped_routes[depot])))
                  <= depots[depot]['capacity'] )
    model += pl.lpSum(C[key]*X[key] for key in X)
    return model, X, C, coverage, trips, grouped_routes

# === RMP Solve & Pricing with on-the-fly dominance ===
def run_rmp(
    lines_info, cp_depot_distances, depots,
    timetables_csv="initializer/files/timetables.csv",
    cplex_path="/Applications/CPLEX_Studio2211/cplex/bin/arm64_osx/cplex",
    log_filename="rmp_output.log"
):
    logger = Logger(log_filename)
    orig_stdout = sys.stdout; sys.stdout = logger
    new_columns = []
    try:
        # solve master
        model, X, C, coverage, trips, grouped = \
            build_rmp_model(lines_info, cp_depot_distances, depots, timetables_csv)
        status = model.solve(pl.CPLEX_CMD(path=cplex_path))
        obj = pl.value(model.objective)
        # extract duals
        dual_alpha, dual_beta = {}, {}
        for name, constr in model.constraints.items():
            if name.startswith("alpha_trip_"): dual_alpha[name[11:]] = constr.pi
            if name.startswith("beta_depot_"): dual_beta[name[12:]] = constr.pi
        print("\n=== Build reduced-cost graph ===")
        G = GraphBuilder(timetables_csv).build_graph()
        for u,v,data in G.edges(data=True):
            ttype = G.nodes[v]['type']; base = data.get('time',0)
            if ttype=='T': rc = base*problem_params['deadhead_cost_per_min'] - dual_alpha.get(v,0)
            elif ttype=='C': rc = 0.0
            else: rc = base*problem_params['deadhead_cost_per_min'] - dual_beta.get(v,0)
            G.edges[u,v]['reduced_cost']=rc
        # pricing with label dominance
        print("\n=== Pricing: label setting SPFA ===")
        for trip in trips:
            labels = spfa_with_dominance(G, trip)
            # collect only sink labels (type K or C)
            for node, lst in labels.items():
                if G.nodes[node]['type'] not in ('K','C'): continue
                for lbl in lst:
                    path, rc, ttime = reconstruct_path_from_label(lbl)
                    if rc < -1e-6:
                        new_columns.append((trip,node,rc,path))
                        print(f"Trip {trip} → {node}, rc={rc:.3f}, time={ttime:.1f}")
        print(f"\nTotal new columns: {len(new_columns)}")
    finally:
        sys.stdout = orig_stdout; logger.log.close()
    return status, model, obj, {'alpha':dual_alpha,'beta':dual_beta}, new_columns

if __name__=='__main__':
    status, model, cost, duals, cols = run_rmp(
        lines_info, cp_depot_distances, depots,
        timetables_csv='initializer/files/timetables.csv',
        cplex_path='/Applications/CPLEX_Studio2211/cplex/bin/arm64_osx/cplex',
        log_filename='rmp_output.log'
    )
    print("Done", cost, cols)
