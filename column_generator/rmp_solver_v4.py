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

    trips = list(pd.read_csv(timetables_csv, usecols=["trip_id"])["trip_id"].values)

    model = pl.LpProblem(name, pl.LpMinimize)
   
    # decision vars & costs
    X = {}
    C = {}

    highest_cost = 0

    for k, depot in enumerate(grouped_routes):
        for p, cost in enumerate(routes_costs[depot]):
            X[(k,p)] = pl.LpVariable(f"{depot}_col{p}", lowBound=0, upBound=1)
            C[(k,p)] = cost
            if cost > highest_cost:
                highest_cost = cost
       
    # coverage matrix
    coverage = {}
    uncovered_trips = []

    for k, (depot, routes) in enumerate(grouped_routes.items()):
        for p, route in enumerate(routes):
            for i, trip in enumerate(trips):
                if trip in route:
                    coverage[(k,p,i)] = 1
    
    # Check which trips are uncovered by regular routes
    for i, trip in enumerate(trips):
        is_covered = False
        for k, depot in enumerate(grouped_routes):
            for p in range(len(grouped_routes[depot])):
                if coverage.get((k,p,i), 0) == 1:
                    is_covered = True
                    break
            if is_covered:
                break
        if not is_covered:
            uncovered_trips.append(trips.index(trip))
    
    if len(uncovered_trips) > 0:
        print(f"Uncovered trips found")
        print(f"All trips will be covered also by an artificial variable.")
        for _p, trip in enumerate(trips):
            X[('NONE', f'ARTIF_{_p}')] = pl.LpVariable(f"ARTIF_{_p}", lowBound=0, upBound=1)
            C[('NONE', f'ARTIF_{_p}')] = highest_cost * 10
            coverage[("NONE", f"ARTIF_{_p}", trip)] = 1
                
    # alpha constraints - ensuring each trip is covered exactly once
    for i, trip in enumerate(trips):
        # Build the constraint including both regular routes AND artificial variable
        constraint_terms = []
        
        # Add terms for regular routes that cover this trip
        for k, depot in enumerate(grouped_routes):
            for p in range(len(grouped_routes[depot])):
                if coverage.get((k,p,i), 0) == 1:
                    constraint_terms.append(X[(k,p)])

        constraint_terms.append(X[("NONE", f"ARTIF_{i}")])

        model += (
            pl.lpSum(constraint_terms) == 1,
            f"alpha_trip_{trip}"
        )

    # beta constraints - depot capacity
    for k, depot in enumerate(grouped_routes):
        model += (
            pl.lpSum(X[(k,p)] for p in range(len(grouped_routes[depot])))
            <= depots[depot]["capacity"],
            f"beta_depot_{depot}"
        )

    # objective function
    model += pl.lpSum(C[key] * X[key] for key in X)

    # Write model file
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
    
    if status == pl.LpStatusOptimal:
        for name, constr in model.constraints.items():
            if name.startswith("alpha_trip_"):
                trip = name.replace("alpha_trip_","")
                alpha[trip] = constr.pi
            elif name.startswith("beta_depot_"):
                depot = name.replace("beta_depot_","")
                beta[depot] = constr.pi
            else:
                # This case might not be an error, could be other constraints
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

    # Check if solution was found
    if status != pl.LpStatusOptimal:
        print(f"\nWARNING: Solution status is {pl.LpStatus[status]}")
        if obj is None:
            print("No objective value found. Model may be infeasible.")
            # Try to debug which trips might be problematic
            print("\nChecking constraint structure...")
            for name, constr in model.constraints.items():
                if name.startswith("alpha_trip_"):
                    # Check if constraint has variables
                    if len(constr) == 0:
                        print(f"  - Constraint {name} has no variables!")
            return status, model, None, {}
    
    print(f"\nZ (Objective Value): {obj:.4f}")

    # Check if artificial variable is used
    artif_var = X.get(('NONE', 'ARTIF'))
    if artif_var and artif_var.varValue and artif_var.varValue > 0.001:
        print(f"\nWARNING: Artificial variable is used with value {artif_var.varValue:.4f}")
        print("This means some trips couldn't be covered by regular routes.")

    # duals
    print("\n--- Dual Values ---")
    if duals.get("alpha"):
        alpha_values = [(trip, pi) for trip, pi in duals["alpha"].items() if pi is not None]
        if alpha_values:
            print("Sample of alpha dual values (first 10):")
            for trip, pi in alpha_values[:10]:
                print(f"  Alpha for trip {trip}: {pi:.4f}")
            if len(alpha_values) > 10:
                print(f"  ... and {len(alpha_values) - 10} more")
    
    if duals.get("beta"):
        for depot, pi in duals["beta"].items():
            if pi is not None:
                print(f"Beta for depot {depot}: {pi:.4f}")

    print("\n--- Routes main data (non-zero values only) ---")
    non_zero_vars = []
    for (k,p), var in X.items():
        var_value = var.varValue if var.varValue is not None else 0.0
        if var_value > 0.001:  # Only show non-zero values
            non_zero_vars.append(((k,p), var, var_value))
    
    # Sort by value for better readability
    non_zero_vars.sort(key=lambda x: x[2], reverse=True)
    
    for (k,p), var, var_value in non_zero_vars:
        print(f"X[({k}, {p})]: {var.name:<15} | Value: {var_value:<10.4f} | Cost: {C.get((k,p), 0.0):.4f}")

    return status, model, obj, duals