path_to_cplex = r'/Applications/CPLEX_Studio2211/cplex/bin/arm64_osx/cplex'
import pulp as pl
model = pl.LpProblem("Example", pl.LpMinimize)
solver = pl.CPLEX_CMD(path=path_to_cplex)
# _var = pl.LpVariable('a')
# _var2 = pl.LpVariable('a2')
# model += _var + _var2 == 1
# result = model.solve(solver)