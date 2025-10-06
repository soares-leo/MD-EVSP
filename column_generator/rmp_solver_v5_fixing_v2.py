import sys
import pulp as pl
import pandas as pd
from initializer.utils import group_routes_by_depot
import numpy as np


def build_rmp_model(
    depots: dict,
    initial_solution: dict,
    timetables_csv: str = "initializer/files/timetables.csv",
    name: str = "RestrictedMasterProblem",
    fixed_vars=None
):
    """
    Build the Restricted Master Problem model.
    
    Args:
        depots: Dictionary of depot information
        initial_solution: Dictionary of initial routes
        timetables_csv: Path to timetables CSV file
        name: Model name
        fixed_vars: List of route keys to suggest as initial solution (warm start)
    
    Returns:
        Tuple of (model, X, C, coverage, trips, grouped_routes, var_to_route_mapping)
    """
    if fixed_vars is None:
        fixed_vars = []
    
    grouped_routes, routes_costs = group_routes_by_depot(initial_solution)
    trips = list(pd.read_csv(timetables_csv, usecols=["trip_id"])["trip_id"].values)
    model = pl.LpProblem(name, pl.LpMinimize)
   
    # Decision variables & costs
    X = {}
    C = {}
    
    # Create mapping from RMP variable names to original route keys
    var_to_route_mapping = {}
    highest_cost = 0

    # Create variables and set warm start values
    for k, depot in enumerate(grouped_routes):
        for p, cost in enumerate(routes_costs[depot]):
            _var = f"{depot}_col{p}"
            
            # Find the original stable key for this temporary variable
            route_path = grouped_routes[depot][p]
            original_route_key = None
            for orig_key, orig_data in initial_solution.items():
                if orig_data["Path"] == route_path:
                    original_route_key = orig_key
                    break
            
            var_to_route_mapping[_var] = original_route_key
            
            # Create variable with standard binary bounds
            X[(k, p)] = pl.LpVariable(_var, lowBound=0, upBound=1)
            
            # Set warm start value (initial hint to solver)
            if original_route_key in fixed_vars:
                X[(k, p)].setInitialValue(1)  # Suggest this route should be used
            else:
                X[(k, p)].setInitialValue(0)  # Suggest this route should not be used
            
            C[(k, p)] = cost
            if cost > highest_cost:
                highest_cost = cost
       
    # Build coverage matrix
    coverage = {}
    uncovered_trips = []

    for k, (depot, routes) in enumerate(grouped_routes.items()):
        for p, route in enumerate(routes):
            for i, trip in enumerate(trips):
                if trip in route:
                    coverage[(k, p, i)] = 1
    
    # Check which trips are uncovered by regular routes
    for i, trip in enumerate(trips):
        is_covered = False
        for k, depot in enumerate(grouped_routes):
            for p in range(len(grouped_routes[depot])):
                if coverage.get((k, p, i), 0) == 1:
                    is_covered = True
                    break
            if is_covered:
                break
        if not is_covered:
            uncovered_trips.append(i)
    
    # Create artificial variables for uncovered trips
    if len(uncovered_trips) > 0:
        print(f"Uncovered trips found: {len(uncovered_trips)} trips")
        print(f"Creating artificial variables for uncovered trips.")
        for i in uncovered_trips:
            artif_var_name = f"ARTIF_{i}"
            X[('NONE', artif_var_name)] = pl.LpVariable(
                artif_var_name, 
                lowBound=0, 
                upBound=1
            )
            # Artificial variables get high cost (penalty)
            C[('NONE', artif_var_name)] = highest_cost * 10
            coverage[("NONE", artif_var_name, i)] = 1
            # Set warm start to 0 for artificial variables
            X[('NONE', artif_var_name)].setInitialValue(0)
                
    # Alpha constraints - ensuring each trip is covered exactly once
    for i, trip in enumerate(trips):
        constraint_terms = []
        
        # Add terms for regular routes that cover this trip
        for k, depot in enumerate(grouped_routes):
            for p in range(len(grouped_routes[depot])):
                if coverage.get((k, p, i), 0) == 1:
                    constraint_terms.append(X[(k, p)])

        # Add artificial variable if it exists for this trip
        if i in uncovered_trips:
            constraint_terms.append(X[("NONE", f"ARTIF_{i}")])

        # Create constraint
        if constraint_terms:
            model += (
                pl.lpSum(constraint_terms) == 1,
                f"alpha_trip_{trip}"
            )
        else:
            print(f"WARNING: Trip {trip} (index {i}) has no coverage!")

    # Beta constraints - depot capacity
    for k, depot in enumerate(grouped_routes):
        model += (
            pl.lpSum(X[(k, p)] for p in range(len(grouped_routes[depot])))
            <= depots[depot]["capacity"],
            f"beta_depot_{depot}"
        )

    # Objective function - minimize total cost
    model += pl.lpSum(C[key] * X[key] for key in X)

    # Write model to file for debugging
    with open(f"model.txt", "w") as f:
        f.write(model.__str__())

    return model, X, C, coverage, trips, grouped_routes, var_to_route_mapping


def solve_rmp_model(
    model: pl.LpProblem,
    cplex_path: str = "/Applications/CPLEX_Studio2211/cplex/bin/arm64_osx/cplex"
):
    """
    Solve the RMP model using CPLEX with warm start enabled.
    
    Args:
        model: The PuLP model to solve
        cplex_path: Path to CPLEX executable
    
    Returns:
        Tuple of (status, objective_value, duals, column_values)
    """
    # Enable warm start in CPLEX solver
    solver = pl.CPLEX_CMD(mip=False, path=cplex_path, timeLimit=300, warmStart=True)
    
    status = model.solve(solver)
    obj = pl.value(model.objective)

    alpha = {}
    beta = {}
    column_values = {}
    
    if status == pl.LpStatusOptimal:
        # Extract dual values (shadow prices)
        for name, constr in model.constraints.items():
            if name.startswith("alpha_trip_"):
                trip = name.replace("alpha_trip_", "")
                alpha[trip] = constr.pi
            elif name.startswith("beta_depot_"):
                depot = name.replace("beta_depot_", "")
                beta[depot] = constr.pi
            else:
                print(f"Constraint '{name}' has dual value: {constr.pi}")

        # Extract primal values (variable values)
        for var in model.variables():
            column_values[var.name] = var.varValue

    return status, obj, {"alpha": alpha, "beta": beta}, column_values


def run_rmp(
    depots: dict,
    initial_solution: dict,
    timetables_csv: str = "initializer/files/timetables.csv",
    cplex_path: str = "/Applications/CPLEX_Studio2211/cplex/bin/arm64_osx/cplex",
    verbose=False,
    fixed_vars=None
):
    """
    Build and solve the Restricted Master Problem.
    
    Args:
        depots: Dictionary of depot information
        initial_solution: Dictionary of initial routes
        timetables_csv: Path to timetables CSV file
        cplex_path: Path to CPLEX executable
        verbose: Whether to print detailed output
        fixed_vars: List of route keys to use as warm start (initial solution hint)
    
    Returns:
        Tuple of (status, model, objective, duals, values, var_to_route_mapping)
    """
    if fixed_vars is None:
        fixed_vars = []
    
    # Build the model with warm start values
    model, X, C, coverage, trips, grouped, var_to_route_mapping = build_rmp_model(
        depots, 
        initial_solution, 
        timetables_csv, 
        "RestrictedMasterProblem", 
        fixed_vars
    )

    print()
    print("-" * 50)
    print("RMP SOLVING RESULTS")
    print("-" * 50)

    # Solve the model
    status, obj, duals, values = solve_rmp_model(model, cplex_path)

    # Check solution status
    if status != pl.LpStatusOptimal:
        print(f"\nWARNING: Solution status is {pl.LpStatus[status]}")
        if obj is None:
            print("No objective value found. Model may be infeasible.")
            print("\nChecking constraint structure...")
            for name, constr in model.constraints.items():
                if name.startswith("alpha_trip_"):
                    if len(constr) == 0:
                        print(f"  - Constraint {name} has no variables!")
            return status, model, None, {}, {}, var_to_route_mapping
    
    print(f"\nZ (Objective Value): {obj:.4f}")

    # Check if artificial variables are used
    artif_used = []
    for key, var in X.items():
        if key[0] == 'NONE' and var.varValue and var.varValue > 0.001:
            artif_used.append((key[1], var.varValue))
    
    if artif_used:
        print(f"\nWARNING: {len(artif_used)} artificial variable(s) are used.")
        if verbose:
            for artif_name, value in artif_used[:5]:
                print(f"  - {artif_name}: {value:.4f}")
            if len(artif_used) > 5:
                print(f"  ... and {len(artif_used) - 5} more")
        print("This means some trips couldn't be covered by regular routes.")

    # Print dual values
    if verbose:
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

    # Collect non-zero variables
    non_zero_vars = []
    for (k, p), var in X.items():
        var_value = var.varValue if var.varValue is not None else 0.0
        if var_value > 0.001:
            non_zero_vars.append(((k, p), var, var_value))
    
    # Sort by value for readability
    non_zero_vars.sort(key=lambda x: x[2], reverse=True)
    
    if verbose:
        print("\n--- Routes main data (non-zero values only) ---")
        for (k, p), var, var_value in non_zero_vars:
            cost = C.get((k, p), 0.0)
            print(f"X[({k}, {p})]: {var.name:<15} | Value: {var_value:<10.4f} | Cost: {cost:.4f}")

    return status, model, obj, duals, values, var_to_route_mapping