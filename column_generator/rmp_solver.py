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
    depots: dict,
    initial_solution: dict,
    timetables_csv: str = "initializer/files/timetables.csv",
    name: str = "RestrictedMasterProblem"
):
    """
    Build the RMP model and return:
      model, X_vars, cost_map, coverage_map, trips_array, grouped_routes
    """

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
    depots: dict,
    initial_solution: dict,
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
            depots, initial_solution, timetables_csv
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

# from initializer.inputs import lines_info, cp_depot_distances, depots

# gen = Generator(lines_info, cp_depot_distances, depots)
# initial_solution, _ = gen.generate_initial_set()

# status, model, optimal_cost, duals = run_rmp(
#     depots,
#     initial_solution,
#     timetables_csv="initializer/files/timetables.csv",
#     cplex_path="/Applications/CPLEX_Studio2211/cplex/bin/arm64_osx/cplex",
#     log_filename="rmp_output.log"
# )

# print("Done. Results also in rmp_output.log")