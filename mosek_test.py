from mosek.fusion import Model, Domain, ObjectiveSense, Expr
with Model() as M:
    x = M.variable(1, Domain.greaterThan(1.0))
    M.objective(ObjectiveSense.Minimize, Expr.sum(x))
    M.solve()
    print('license ok, x =', x.level())