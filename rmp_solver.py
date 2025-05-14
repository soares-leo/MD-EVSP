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


from collections import deque

def run_spfa(graph, source_node, D, T_d):
    red_costs = {source_node: 0}
    red_costs.update({n: math.inf for n in graph.nodes if n != source_node})
    dist = {source_node: 0}
    dist.update({n: math.inf for n in graph.nodes if n != source_node})
    time = {source_node: 0}
    time.update({n: math.inf for n in graph.nodes if n != source_node})
    queue = deque()
    queue.append(source_node)
    iteration=1
    while len(queue) > 0:
        u = queue.popleft()
        print(f"Iteration: {iteration}. Queue len: {len(queue)}. U: {u}.")
        current_time = graph.nodes[u]["start_time"]
        start_time = graph.nodes[u]["start_time"]
        planned_travel_time = graph.nodes[u]["planned_travel_time"]
        end_time = graph.nodes[u]["end_time"]
        start_cp = graph.nodes[u]['start_cp']
        end_cp = graph.nodes[u]['end_cp']

        for v in graph.neighbors(u):
            if v[0] != "l":
                continue
            arc_reduced_cost = graph[u][v]["reduced_cost"]
            arc_time = graph[u][v]["time"]
            arc_dist = graph[u][v]["dist"]
            if time[u] + arc_time >= T_d:
                continue
            elif dist[u] + arc_dist > D:
                cs = "c + numero do deposito prox do end cp de u"
                arc_reduced_cost = graph[u][cs]["reduced_cost"]
                arc_dist = graph[u][cs]["dist"]
                time_since_recharge = ((dist[u] + arc_dist) / 20) * 60
                charging_time = 15.6 * time_since_recharge / 30
                arc_time = graph[u][cs]["time"] + charging_time
                red_costs[v] = red_costs[u] + arc_reduced_cost
                #precisa ter contabilizador de dist que possa ser zerado.
                dist[v]

                

            if red_costs[u] + arc_reduced_cost < red_costs[v] and time[u] + arc_time < T_d:
                if time[u] + arc_time >= T_d:
                    return red_costs, time, dist
                elif dist[u] + arc_dist > D:

                red_costs[v] = red_costs[u] + arc_reduced_cost
                if v not in queue:
                    queue.append(v)
        iteration+=1
    return red_costs

trip_nodes = [n for n, attr in graph.nodes(data=True) if attr.get("type") == "T"]
for i in trip_nodes:
    red_costs = run_spfa(graph, i)
    print(red_costs)




# para cada K:
    # para cada nó i \in T:
        # SPFA, ensuring dist and time. 
            # OBS: na regra de dominancia, o nó que cobre maior distância e tem menor custo, prevalece.
            # OBS2: como ele nao recebe dest_node, ele acaba achando sp para todos os nós.
            # OBS3: restricao de capacidade de depositos ja é resolvida no RMP.