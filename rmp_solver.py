import pulp as pl
from initializer.inputs import *
from initializer.generator import Generator
from initializer.utils import group_routes_by_depot
import numpy as np
import pandas as pd
import sys

# Redirect output to both console and a file
class Logger(object):
    def __init__(self, filename="rmp_output.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):  # Required for compatibility
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger("rmp_output.log")

# === Configuration ===
path_to_cplex = "/Applications/CPLEX_Studio2211/cplex/bin/arm64_osx/cplex"
solver = pl.CPLEX_CMD(path=path_to_cplex)
model = pl.LpProblem("RestrictedMasterProblem", pl.LpMinimize)

# === Generate Initial Routes ===
print("="*85)
print("SECTION 1: INITIAL SOLUTION GENERATION PROCESS")
print("="*85)
generator = Generator(lines_info=lines_info, cp_depot_distances=cp_depot_distances, depots=depots)
initial_solution, used_depots = generator.generate_initial_set()
grouped_routes, routes_costs = group_routes_by_depot(initial_solution)

# === Read Trips ===
trips = pd.read_csv("initializer/files/timetables.csv", usecols=["trip_id"]).trip_id.values

# === Decision Variables ===
X = {}  # Route selection variables
C = {}  # Route costs

for i, (depot, routes) in enumerate(grouped_routes.items()):
    for j in range(len(routes)):
        X[i, j] = pl.LpVariable(f"{depot}_col{j}", lowBound=0, upBound=1, cat="Continuous")

for i, (depot, costs) in enumerate(routes_costs.items()):
    for j, cost in enumerate(costs):
        C[i, j] = cost

# === Trip Coverage Constraints (α duals) ===
constraint_2_trips_values = {}
for k, (depot, routes) in enumerate(grouped_routes.items()):
    for p, route in enumerate(routes):
        for i, trip in enumerate(trips):
            constraint_2_trips_values[k, p, i] = 1 if trip in route else 0

for i, trip in enumerate(trips):
    model += pl.lpSum(
        constraint_2_trips_values[k, p, i] * X[k, p]
        for k in range(len(grouped_routes))
        for p in range(len(grouped_routes[list(grouped_routes.keys())[k]]))
    ) == 1, f"alpha_trip_{trip}"

# === Depot Capacity Constraints (β duals) ===
for k, depot in enumerate(grouped_routes.keys()):
    model += pl.lpSum(
        X[k, p] for p in range(len(grouped_routes[depot]))
    ) <= depots[depot]["capacity"], f"beta_depot_{depot}"

# === Objective Function ===
model.setObjective(
    pl.lpSum(C[i, j] * X[i, j] for (i, j) in X)
)

# === Solve ===
result = model.solve(solver)

# === Output Solution ===
print()
print("="*85)
print("SECTION 2: RMP SOLVING RESULTS")
print("="*85)
print("\n--- Routes Coefficients ---")
for (i, j), var in X.items():
    print(f"X[{i},{j}] = {var.varValue:.4f} (cost = {C[i, j]})")

print("\n--- Trip Coverage Check ---")
for i, trip in enumerate(trips):
    lhs = sum(
        constraint_2_trips_values[k, p, i] * X[k, p].varValue
        for k in range(len(grouped_routes))
        for p in range(len(grouped_routes[list(grouped_routes.keys())[k]]))
    )
    print(f"Trip {trip}: coverage = {lhs:.4f}")

print(f"\nZ (Objective Value): {pl.value(model.objective):.4f}")

# === Duals ===
print("\n--- Dual Values ---")
for name, constraint in model.constraints.items():
    if name.startswith("alpha_trip_"):
        trip_id = name.replace("alpha_trip_", "")
        print(f"α for trip {trip_id}: {constraint.pi:.4f}")
    elif name.startswith("beta_depot_"):
        depot_id = name.replace("beta_depot_", "")
        print(f"β for depot {depot_id}: {constraint.pi:.4f}")
