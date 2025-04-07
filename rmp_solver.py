
import pulp as pl
from initializer.inputs import *
from initializer.generator import Generator
from initializer.utils import group_routes_by_depot
import numpy as np
import pandas as pd

path_to_cplex = "/Applications/CPLEX_Studio2211/cplex/bin/arm64_osx/cplex"
model = pl.LpProblem("PP", pl.LpMinimize)
solver = pl.CPLEX_CMD(path=path_to_cplex)
generator = Generator(lines_info=lines_info, cp_depot_distances=cp_depot_distances, depots=depots)
initial_solution, used_depots = generator.generate_initial_set()

grouped_routes, routes_costs = group_routes_by_depot(initial_solution)

X = {}
C = {}
constraint_2_trips = {}
constraint_2_trips_values = {}
constraint_3_dep = {}
trips = pd.read_csv("initializer/files/timetables.csv", usecols=["trip_id"]).trip_id.values

for i, v in enumerate(grouped_routes.items()):
    for j in range(0, len(grouped_routes[v[0]])):
        X[i,j] = pl.LpVariable(f"{v[0]}_col{j}", lowBound=0, upBound=1, cat="Continuous")

for i, v in enumerate(routes_costs.items()):
    for j, cost in enumerate(routes_costs[v[0]]):
        C[i,j] = cost

for k, dep in enumerate(grouped_routes.items()):
    for p, col in enumerate(dep[1]):
        for i, trip in enumerate(trips):
            if trip in col:
                a = 1
            else: a = 0
            constraint_2_trips[k, p, i] = pl.LpVariable(name=f"dep_{dep[0]}_col_{p}_{i}", cat="Binary")
            constraint_2_trips_values[k, p, i] = a

for k, dep in enumerate(grouped_routes.keys()):
    for p, route in enumerate(grouped_routes[dep]):
        if dep in route:
            d = 1
        else: d = 0
        constraint_3_dep[k, p] = d

# objective function
model.setObjective(sum(C[i,j]*X[i,j] for i in range(0, len(grouped_routes)) for j in range(0, len(list(grouped_routes.values())[i]))))

# trips coverage constraints
for i in range(len(trips)):
    model += pl.lpSum(
        constraint_2_trips_values[k, p, i] * X[k, p]
        for k in range(len(grouped_routes))
        for p in range(len(list(grouped_routes.values())[k]))
    ) == 1

# depots capacity constraints
for k, dep in enumerate(grouped_routes.keys()):
   model += pl.lpSum(constraint_3_dep[k, p] * X[k,p]
                     for p in range(0, len(list(grouped_routes.values())[k])))  <= depots[dep]["capacity"]

result = model.solve(solver)

for (i, j), var in X.items():
    if var.varValue > 1e-5:
        print(f"X[{i},{j}] = {var.varValue:.4f} (cost = {C[i,j]})")

for i in range(len(trips)):
    lhs = sum(constraint_2_trips_values[k, p, i] * X[k, p].varValue
              for k in range(len(grouped_routes))
              for p in range(len(list(grouped_routes.values())[k])))
    print(f"Trip {trips[i]}: coverage = {lhs:.4f}")


print(f"Z: {pl.value(model.objective):.4f}")

# duals
for name, constraint in model.constraints.items():
    print(f"Constraint: {name}, Dual value: {constraint.pi:.4f}")




