# implementing pymoo multi-objective optimization with FEniCS FEA objectives

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import pathlib
from pathlib import Path

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize

from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.hux import HalfUniformCrossover
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.decomposition.asf import ASF

# importing required modules (have to be in the same directory!)
from core_functions import *
from boundary_conditions import *


# defining the multiobjective problem for square FEA problem: Weight/vm_stress obj
class MyProblem(ElementwiseProblem):

    def __init__(self, conn, coords, num_el, meshpath):
        super().__init__(n_var=num_el,
                         n_obj=2,
                         n_ieq_constr=0,
                         xl=np.zeros(num_el),
                         xu=np.ones(num_el)
                         )
        self.conn = conn
        self.coords = coords
        self.num_el = num_el
        self.meshpath = meshpath  # temp mesh for evaluation

    def _evaluate(self, x, out, *args, **kwargs):

        # generate the infill file
        generate_infill_file(self.meshpath, x, self.conn, self.coords)

        # Loading the created infill mesh to FEniCS
        mesh, _, _, _ = extract_mesh_data_fenics(meshfile)

        # perform FEA and obtain displacements and von mises stress
        d_max, d_tot, d_avg, max_vm = square_FEA(mesh)

        # objective values: we want to minimize *weight* and *maximal stress*
        f1 = np.sum(x)  # objective 1 - "weight"
        f2 = d_max     # objective 2 - "maxStress"

        # constraints
        #g1 = np.sum(x) - 200   # dummy constraint: can be adjusted to limit the infill rate

        out["F"] = [f1, f2]  # dictionary key for objectives
        #out["G"] = [g1]      # dictionary key for constraints


# Filepath management
cwd = Path.cwd()
mesh_dir = cwd.joinpath("MOEA_meshes", "smallSquare7x7")  # adjust the dir here
inf_file = str(mesh_dir.stem) + '-infill.msh'
mesh_inp = mesh_dir / inf_file  # mesh input file

# Extracting geometry from the infill mesh -> num_el is the design dimension
conn, coords, num_el = extract_mesh_data(mesh_inp)   # extracting the geometry

# init the MOP
meshfile = mesh_dir / 'infill_gen.msh'
problem = MyProblem(conn, coords, num_el, meshfile)


# initialize the algorithm object
algorithm = NSGA2(
    pop_size=100,
    n_offsprings=40,
    sampling=BinaryRandomSampling(),
    crossover=TwoPointCrossover(prob=1.0),
    #crossover=UniformCrossover(),
    mutation=BitflipMutation(prob=0.5, prob_var= 2 * 1/num_el),  # mutation scaled by the num of elements
    eliminate_duplicates=True
)


# define the termination criterion
termination = get_termination("n_gen", 100)


# solve the MOP
res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)

# n_gen  - generation counter,
# n_eval - number of evaluations
# cv_min - minimum constraint violation
# cv_avg - average constraint violation in the current population.
# n_nds  - number of non-dominated solutions
# eps/ind- running performance metrics

X = res.X  # solutions in the decision space
F = res.F  # solutions in the objective space
hist = res.history  # getting the info for each iteration

#print(X)
#print(F)

# save solutions
out_dir = mesh_dir / "infill_gen"
img_dir = out_dir / "png"
out_dir.mkdir(exist_ok=True)
img_dir.mkdir(exist_ok=True)
for i, x in enumerate(X):
    ifile = out_dir / f"sol{i:02d}.msh"  # generate mesh
    generate_infill_file(ifile, x, conn, coords)
    imfile = img_dir / f"sol_{i}.png"    # generate png
    generate_img_file(imfile, x, conn, coords)
    print(f"File {ifile.name} saved.")###

# writing to a file
with open(out_dir / "sq_solutions.txt", "w+") as file:
    alg = str(type(algorithm))[8:-2]
    header = f"Algorithm:{alg}, pop_size: {algorithm.pop_size},\
     n_offsp: {algorithm.n_offsprings}, n_gen: {termination.n_max_gen},\
     t_tot: {res.exec_time:10.1f}, t_gen: {res.exec_time/termination.n_max_gen:10.2f}\n"
    file.writelines(header)
    file.writelines(f"{str(list(sol))} \t w: {F[i][0]:10.1f} \t vm: {F[i][1]:10.06f} \n" for i, sol in enumerate(1*X))

# plotting the decision space
xl, xu = problem.bounds()
plt.figure(figsize=(7, 5))
plt.imshow(X)
plt.title("Design Space")
plt.show()


# plotting the objective space
plt.figure(figsize=(5, 3))
plt.scatter(F[:, 0], F[:, 1], s=20, facecolors='black', edgecolors='black')
plt.title("Objective Space")
plt.savefig(out_dir / 'ObjectiveSpace.png')
plt.show()


# Multi-Criteria Decision Making: subset selection
n_keep = 9  # how many solutions should we keep
if len(X) > n_keep:
    approx_ideal = F.min(axis=0)  # ideal point
    approx_nadir = F.max(axis=0)  # nadir point

    # normalizing with respect to both objectives
    nF = (F - approx_ideal) / (approx_nadir - approx_ideal)  # normalized objectives

    # Decision making: Using Augmented Scalarization Function (ASF)
    decomp = ASF()                  # init the ASF metric to be minimized
    weights = [np.array([(k + 1)/(n_keep+1), (n_keep - k)/(n_keep+1)]) for k in range(n_keep)]
    best_ASF = []  # indexes of chosen solutions
    for weight in weights:
        best_ASF.append(decomp.do(nF, 1/weight).argmin())  # index of the best solution regarding ASF
    best_ASF = list(set(best_ASF))  # remove duplicates
    F_ASF = F[best_ASF, :]          # objectives
    X_ASF = X[best_ASF, :]          # solutions
    n_kept = len(best_ASF)          # number of kept solutions


    # plotting the objective space with mesh png annotations
    fig, ax = plt.subplots(figsize=(13, 8))
    plt.scatter(F_ASF[:, 0], F_ASF[:, 1], s=20, facecolors='black', edgecolors='black')

    for i, (x, y) in enumerate(F_ASF):
        ind = best_ASF[i]
        img_lab = plt.imread(img_dir / f"sol_{ind}.png")
        imagebox = OffsetImage(img_lab, zoom=0.045)
        imagebox.image.axes = ax

        fl_offset = (-1)**i * 20  # spreading fluctuating offset
        ab = AnnotationBbox(imagebox, (x, y),
                            xybox=(32 - 0.6 * fl_offset, 38 + fl_offset),
                            xycoords='data',
                            boxcoords=("offset points", "offset points"),
                            pad=-1,
                            arrowprops=dict(
                                arrowstyle="->",
                                connectionstyle="angle,angleA=0,angleB=90,rad=3",
                                linewidth=0.5)
                            )
        ax.add_artist(ab)
        ax.text(x, y, f"  {str(ind)}", va='top', ha='left', zorder=4)

    xrang = np.abs(np.max(F_ASF[:,0]) - np.min(F_ASF[:,0]))
    yrang = np.abs(np.max(F_ASF[:,1]) - np.min(F_ASF[:,1]))
    ax.set_xlim((np.min(F_ASF[:,0]) - 0.05*xrang, np.max(F_ASF[:,0]) + 0.15*xrang ))
    ax.set_ylim((np.min(F_ASF[:,1]) - 0.1*yrang, np.max(F_ASF[:,1]) + 0.25*yrang ))
    plt.title(f"Objective Space of {n_kept} ASF-selected solutions")
    plt.savefig(out_dir / 'ObjectiveSpace-ASF.png')
    plt.show()
