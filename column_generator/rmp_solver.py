import sys
import pulp as pl
import pandas as pd
from initializer.generator import Generator
from initializer.utils import group_routes_by_depot


def build_rmp_model(
    depots: dict,
    initial_solution: dict,
    timetables_csv: str = "initializer/files/timetables.csv",
    name: str = "RestrictedMasterProblem"
):

    grouped_routes, routes_costs = group_routes_by_depot(initial_solution)

    trips = pd.read_csv(timetables_csv, usecols=["trip_id"])["trip_id"].values

    model = pl.LpProblem(name, pl.LpMinimize)

    # decision vars & costs
    X = {}
    C = {}
    for k, depot in enumerate(grouped_routes):
        for p, cost in enumerate(routes_costs[depot]):
            X[(k,p)] = pl.LpVariable(f"{depot}_col{p}", lowBound=0,)
            C[(k,p)] = cost
    
    # X[('NONE', 'ARTIF')] = pl.LpVariable("ARTIF", lowBound=0)
    # C[('NONE', 'ARTIF')] = 100000

    # coverage matrix
    coverage = {}
    for k, (depot, routes) in enumerate(grouped_routes.items()):
        for p, route in enumerate(routes):
            for i, trip in enumerate(trips):
                coverage[(k,p,i)] = int(trip in route)

    # alpha constraints
    for i, trip in enumerate(trips):
        model += (
            pl.lpSum(coverage[(k,p,i)] * X[(k,p)]
                     for k in range(len(grouped_routes))
                     for p in range(len(grouped_routes[list(grouped_routes.keys())[k]])))
            # + X[('NONE', 'ARTIF')]
            == 1,
            f"alpha_trip_{trip}"
        )

    # beta constraints
    for k, depot in enumerate(grouped_routes):
        model += (
            pl.lpSum(X[(k,p)] for p in range(len(grouped_routes[depot])))
            <= depots[depot]["capacity"],
            f"beta_depot_{depot}"
        )

    # objective function
    model += pl.lpSum(C[key] * X[key] for key in X)

    return model, X, C, coverage, trips, grouped_routes

def solve_rmp_model(
    model: pl.LpProblem,
    cplex_path: str = "/Applications/CPLEX_Studio2211/cplex/bin/arm64_osx/cplex"
):

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
        else:
            print("############################# HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(constr.pi)

    return status, obj, {"alpha": alpha, "beta": beta}

# === Wrapper that keeps your prints + logger ===
def run_rmp(
    depots: dict,
    initial_solution: dict,
    timetables_csv: str = "initializer/files/timetables.csv",
    cplex_path: str = "/Applications/CPLEX_Studio2211/cplex/bin/arm64_osx/cplex"
):

    model, X, C, coverage, trips, grouped = build_rmp_model(
        depots, initial_solution, timetables_csv
    )

    print()
    print("-"*50)
    print("RMP SOLVING RESULTS")
    print("-"*50)

    status, obj, duals = solve_rmp_model(model, cplex_path)

    print(f"\nZ (Objective Value): {obj:.4f}")

    # coverage check
    # print("\n--- Trip Coverage Check ---")
    # for i, trip in enumerate(trips):
    #     lhs = sum(coverage[(k,p,i)] * X[(k,p)].varValue
    #               for k in range(len(grouped))
    #               for p in range(len(grouped[list(grouped.keys())[k]])))
    #     print(f"Trip {trip}: coverage = {lhs:.4f}")

    # duals
    print("\n--- Dual Values ---")
    for trip, π in duals["alpha"].items():
        print(f"Alpha for trip {trip}: {π:.4f}")
    for depot, π in duals["beta"].items():
        print(f"Beta for depot {depot}: {π:.4f}")


    print("\n--- Routes main data ---")
    for v, ((k,p), var) in zip(model.variables(), X.items()):
        width = len(v.name)
        blanks = 14 - width
        width_p = len(str(p))
        blanks_p = 5 - width_p
        print(f"X[{k},{blanks_p*' '}{p}]  ({v.name}){blanks*' '}|   Coef (decision var): {v.varValue:.4f}   |   Cost: {C[(k,p)]:.4f}")

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

#157008.5048