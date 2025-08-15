import sys
import pulp


# === Logger class ===
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# === RMP solver ===
def solve_rmp(cost_arr, time_arr, cplex_path):
    var_keys = list(range(len(cost_arr)))
    rmp = pulp.LpProblem("SP_RMP", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", var_keys, lowBound=0)

    rmp += pulp.lpSum(cost_arr[i] * x[i] for i in var_keys)
    rmp += pulp.lpSum(time_arr[i] * x[i] for i in var_keys) <= 14
    rmp += pulp.lpSum(x[i] for i in var_keys) == 1

    logger = Logger("rmp_output.log")
    original_stdout = sys.stdout
    sys.stdout = logger

    try:
        print("\nRestricted Master Problem (RMP) Data:")
        print(rmp)
        solver = pulp.CPLEX_CMD(path=cplex_path, msg=1)
        rmp.solve(solver)
    finally:
        sys.stdout = original_stdout
        logger.log.close()

    objective = pulp.value(rmp.objective)
    solution = [var.varValue for var in rmp.variables()]
    duals = [cons.pi for cons in rmp.constraints.values()]

    return {
        'objective': objective,
        'solution': solution,
        'duals': duals
    }


# === SP solver ===
def solve_sp(A, c, t, pi_1, pi_0, cplex_path):
    sp = pulp.LpProblem("SP_SP", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", A, cat='Binary')

    cost = {arc: c[i] - pi_1 * t[i] for i, arc in enumerate(A)}
    sp += pulp.lpSum([cost[arc] * x[arc] for arc in A]) - pi_0

    sp += pulp.lpSum([x[(1, j)] for (u, j) in A if u == 1]) == 1
    for i in [2, 3, 4, 5]:
        sp += pulp.lpSum([x[(i, j)] for (u, j) in A if u == i]) - pulp.lpSum([x[(j, i)] for (j, u) in A if u == i]) == 0
    sp += pulp.lpSum([x[(i, 6)] for (i, u) in A if u == 6]) == 1

    logger = Logger("sp_output.log")
    original_stdout = sys.stdout
    sys.stdout = logger

    try:
        print("\nShortest Path (SP) Data:")
        print(sp)
        solver = pulp.CPLEX_CMD(path=cplex_path, msg=1)
        sp.solve(solver)
    finally:
        sys.stdout = original_stdout
        logger.log.close()

    objective = pulp.value(sp.objective)
    solution = [var.varValue for var in sp.variables()]
    total_c = sum([a * b for a, b in zip(c, solution)])
    total_t = sum([a * b for a, b in zip(t, solution)])

    return {
        'objective': objective,
        'solution': solution,
        'total_c': total_c,
        'total_t': total_t
    }


# === Main loop ===
if __name__ == "__main__":
    A = [(1,2), (1,3), (2,4), (2,5), (3,2), (3,4), (3,5), (4,5), (4,6), (5,6)]
    c = [1, 10, 1, 2, 1, 5, 12, 10, 1, 2]
    t = [10, 3, 1, 3, 2, 7, 3, 1, 7, 2]
    cplex_path = "C:/Program Files/IBM/ILOG/CPLEX_Studio2211/cplex/bin/x64_win64/cplex.exe"

    c_coefs = [100]
    t_coefs = [0]
    rmp_dict = None
    sp_dict = None
    generated_paths = {}
    objective = -1
    iteration = 1

    while objective < 0:
        print(f"{'*'*30} ITERATION {iteration} {'*'*30}\n")

        print("STEP 1: SOLVING RMP")
        rmp_dict = solve_rmp(c_coefs, t_coefs, cplex_path)

        print("RMP Solution:")
        for k, v in rmp_dict.items():
            print(f'{k}: {v}')
        print()

        pi_1 = rmp_dict['duals'][0]
        pi_0 = rmp_dict['duals'][1]

        print("pi_1:", pi_1)
        print("pi_0:", pi_0)
        print()

        try:
            arcs = [A[i] for i, a in enumerate(sp_dict['solution']) if a == 1]
            root = min(arcs, key=lambda x: x[0])
            path = [root]
            _arcs = arcs.copy()
            _arcs.remove(root)
            while _arcs:
                for arc in _arcs:
                    if arc[0] == path[-1][1]:
                        path.append(arc)
                        _arcs.remove(arc)
            path = [path[0][0]] + [path[0][1]] + list(map(lambda x: x[1], path[1:]))
            generated_paths[f"Iteration {iteration}"] = path
        except:
            path = "Nenhum"
            generated_paths[f"Iteration {iteration}"] = path

        print("Generated Path:", path)
        print()

        print("STEP 2: SOLVING SP")
        sp_dict = solve_sp(A, c, t, pi_1, pi_0, cplex_path)

        print("SP Solution:")
        for k, v in sp_dict.items():
            print(f'{k}: {v}')
        print()

        objective = sp_dict['objective']
        c_coefs.append(sp_dict['total_c'])
        t_coefs.append(sp_dict['total_t'])

        print("Updated RMP Coefficients:")
        print("c:", c_coefs)
        print("t:", t_coefs)
        print()

        iteration += 1

    print("-" * 100)
    print("\nGenerated Paths:")
    for k, v in generated_paths.items():
        print(f'{k}: {v}')
