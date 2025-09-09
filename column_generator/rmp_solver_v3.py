import sys
import pulp as pl
import pandas as pd
from initializer.utils import group_routes_by_depot
import numpy as np


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

    highest_cost = 0

    for k, depot in enumerate(grouped_routes):
        for p, cost in enumerate(routes_costs[depot]):
            
            X[(k,p)] = pl.LpVariable(f"{depot}_col{p}", lowBound=0,)
            C[(k,p)] = cost

            if cost > highest_cost:
                highest_cost = cost
   
    # coverage matrix
    coverage = {}
    uncovered_trips = trips.copy()

    for k, (depot, routes) in enumerate(grouped_routes.items()):
        for p, route in enumerate(routes):
            for i, trip in enumerate(trips):
                if trip in route:
                    coverage[(k,p,i)] = 1
                    # Use a boolean mask for more efficient removal
                    if trip in uncovered_trips:
                        uncovered_trips = uncovered_trips[uncovered_trips != trip]

    uncovered_trips_indexes = []

    if len(uncovered_trips) > 0:     
        print(f"Uncovered trips: {uncovered_trips}")

        X[('NONE', 'ARTIF')] = pl.LpVariable("ARTIF", lowBound=0)
        C[('NONE', 'ARTIF')] = highest_cost * len(uncovered_trips) / 8 * 100

        uncovered_trips_indexes = [np.where(trips == trip)[0][0] for trip in uncovered_trips]

        for i in uncovered_trips_indexes:
            coverage[("NONE","ARTIF",i)] = 1

            model += (
                pl.lpSum(
                    coverage[("NONE", "ARTIF", i)] * X[("NONE", "ARTIF")])
                == 1,
                f"alpha_trip_{trips[i]}"
            )

    # alpha constraints
    for i, trip in enumerate(trips):

        # This check now correctly compares an integer 'i' with a list of integer indices.
        if i in uncovered_trips_indexes:
            continue
        else:
            model += (
                pl.lpSum(coverage.get((k,p,i), 0) * X[(k,p)]
                         for k in range(len(grouped_routes))
                         for p in range(len(grouped_routes[list(grouped_routes.keys())[k]])))
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

    with open(f"model.txt", "w") as f:
        f.write(model.__str__())

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
            print(f"Constraint '{name}' has dual value: {constr.pi}")

    return status, obj, {"alpha": alpha, "beta": beta}

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

    # duals
    print("\n--- Dual Values ---")

    if duals["alpha"]:
        for trip, pi in duals["alpha"].items():
            print(f"Alpha for trip {trip}: {pi:.4f}")
    if duals["beta"]:
        for depot, pi in duals["beta"].items():
            print(f"Beta for depot {depot}: {pi:.4f}")

    print("\n--- Routes main data ---")

    for (k,p), var in X.items():
        var_value = var.varValue if var.varValue is not None else 0.0

        print(f"X[({k}, {p})]: {var.name:<15} | Value: {var_value:<10.4f} | Cost: {C.get((k,p), 0.0):.4f}")

    return status, model, obj, duals