# Boundary condition functions setup for FEniCS

# boundary connection setup
tol = 1
def clamped_middle(x, on_boundary): # beam is only fixed in the middle
    #return on_boundary and near(x[0], 0, tol) and near(x[1], 20, tol)#x[0] < tol
    return on_boundary and near(x[0], 0 + 0.25, tol) and near(x[1], 5, tol)#x[0] < tol

# Define strain
def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)
    # return sym(nabla_grad(u))

# Define stress
def sigma(u, lambda_, d, mu):
    return lambda_*nabla_div(u)*Identity(d) + 2*mu*epsilon(u)

# Delta function for point load
class Delta(UserExpression):
    def __init__(self, eps, x0, **kwargs):
        self.eps = eps
        self.x0 = x0
        UserExpression.__init__(self, **kwargs)
    def eval(self, values, x):
        eps = self.eps
        values[0] = eps/pi/(np.linalg.norm(x-self.x0)**2 + eps**2)
        values[1] = 0
    def value_shape(self): return (2, )

# Obtain max, total and average displacements
def get_displacement_measures(u, V):
    u_magnitude = sqrt(dot(u, u))
    u_magnitude = project(u_magnitude, V)
    u_magnitude = u_magnitude.vector()
    disp_len = u_magnitude.size()
    disp_max = u_magnitude.max()   # maximal displacement
    disp_tot = u_magnitude.sum()   # total/sum displacement
    disp_avg = disp_tot / disp_len # average displacement
    return disp_max, disp_tot, disp_avg

# returns displacements and vm stress for side-compression of the square
def square_FEA(mesh, verbose=False, plotting=False):
    if not verbose:
        set_log_active(False)  # dolfin verbose to false
    else:
        set_log_active(True)   # dolfin verbose to true

    # Define function space for system of PDEs
    degree = 2
    lambda_ = 1
    mu = 1
    V = VectorFunctionSpace(mesh, 'P', degree)

    # Define boundary conditions
    bc1 = DirichletBC(V, Constant((0, 0)), clamped_middle)
    bcs = [bc1]  # combined boundary conditions
    delta = Delta(eps=1E-4, x0=np.array([9.75, 5]), degree=5)  # point load at x0

    # Define variational problem
    u = TrialFunction(V)
    d = u.geometric_dimension()   # space dimension
    v = TestFunction(V)
    f = Constant((0, 0))
    T = Constant((0, 0))
    a = inner(sigma(u, lambda_, d, mu), epsilon(v)) * dx
    L = inner(Constant(-10) * delta, v) * dx

    # Compute solution
    u = Function(V)
    solve(a == L, u, bcs)

    # Displacement measures
    V1 = FunctionSpace(mesh, 'P', 1)
    disp_max, disp_tot, disp_avg = get_displacement_measures(u, V1)
    if plotting:
        title_str = f"Displacement \n max:{round(disp_max,3)}  tot:{round(disp_tot,3)}  avg:{round(disp_avg,3)}"
        plot(u, title=title_str, mode='displacement')

    # max Von Mises Stress measure
    s = sigma(u, lambda_, d, mu) - (1./3) * tr(sigma(u, lambda_, d, mu)) * Identity(d)  # deviatoric stress
    von_mises = sqrt(3./2 * inner(s, s))
    von_mises = project(von_mises, V1)
    max_von_mises = von_mises.vector().max()  # getting maximal von Mises stress
    #plot(von_mises, title=f"Stress intensity, max vonMis:{max_von_mises}")###

    return disp_max, disp_tot, disp_avg, max_von_mises
