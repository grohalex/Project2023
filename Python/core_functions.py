# Python-based Implementation of multi-objective optimization of infill
# patterns using pymoo optimization library with structural objectives
# based on FEA evaluation in FEniCS framework with Gmsh meshing

import numpy as np
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2

from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.hux import HalfUniformCrossover
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation

from pymoo.termination import get_termination
from pymoo.optimize import minimize

import meshio
import gmsh
from dolfin import *

def create_mesh(mesh, cell_type):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:geometrical", cell_type)
    out_mesh = meshio.Mesh(points=mesh.points[:,:2], cells={cell_type: cells}, cell_data={"name_to_read": [cell_data]})
    return out_mesh


def extract_mesh_data(mesh_to_load):
    msh = meshio.read(mesh_to_load)

    # filepath management
    cwd = Path.cwd()
    temp_dir = cwd.joinpath("temp_mesh")   # temp dir for loading meshes
    temp_dir.mkdir(exist_ok=True)          # create parent dir if it does not exist


    infill_triangles = create_mesh(msh, "triangle")
    infill_lines = create_mesh(msh, "line")
    meshio.write(temp_dir / "mesh.xdmf", infill_triangles)
    meshio.write(temp_dir / "mf.xdmf", infill_lines)

    infill_mesh = Mesh()
    mvc = MeshValueCollection("size_t", infill_mesh, infill_mesh.topology().dim())
    with XDMFFile(str(temp_dir / "mesh.xdmf")) as infile:
        infile.read(infill_mesh)
        infile.read(mvc, "name_to_read")
    cf = cpp.mesh.MeshFunctionSizet(infill_mesh, mvc)

    mvc = MeshValueCollection("size_t", infill_mesh, infill_mesh.topology().dim() - 1)
    with XDMFFile(str(temp_dir / "mf.xdmf")) as infile:
        infile.read(mvc, "name_to_read")
    mf = cpp.mesh.MeshFunctionSizet(infill_mesh, mvc)

    infill_connect = infill_mesh.cells()       # triangle connectivity of the infill mesh
    infill_coords = infill_mesh.coordinates()  # coordinates within the infill mesh
    infill_coords = np.round(infill_coords, 4) # rounding the coordinates
    num_el = len(infill_connect)               # number of infill elements

    return infill_connect, infill_coords, num_el

def extract_mesh_data_fenics(mesh_to_load):
    msh = meshio.read(mesh_to_load)

    # filepath management
    cwd = Path.cwd()
    temp_dir = cwd.joinpath("temp_mesh")   # temp dir for loading meshes
    temp_dir.mkdir(exist_ok=True)          # create parent dir if it does not exist

    triangle_mesh = create_mesh(msh, "triangle")
    line_mesh = create_mesh(msh, "line")
    meshio.write(temp_dir / "mesh.xdmf", triangle_mesh)
    meshio.write(temp_dir / "mf.xdmf", line_mesh)

    mesh = Mesh()
    mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
    with XDMFFile(str(temp_dir / "mesh.xdmf")) as infile:
        infile.read(mesh)
        infile.read(mvc, "name_to_read")
    cf = cpp.mesh.MeshFunctionSizet(mesh, mvc)

    mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim() - 1)
    with XDMFFile(str(temp_dir / "mf.xdmf")) as infile:
        infile.read(mvc, "name_to_read")
    mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)

    return mesh, mvc, cf, mf

def generate_infill_file(meshpath, infill_vec, infill_connect, infill_coords, verbosity=1, run_GUI=False):
    size = 10  # physical dimension of the side
    P = 1      # number of perimeters
    ex = 0.5   # extrusion width (one perimeter width)
    width, height = size, size       # setting up the square
    num_el_x, num_el_y = 2 * size, 2 * size  # adjust the num of elements
    num_el_x_inf, num_el_y_inf = 7, 7  # adjust num of elems for infill
    dx, dy = width / num_el_x, height / num_el_y

    # initialize Gmsh
    if not gmsh.is_initialized():
        gmsh.initialize()
        gmsh.option.setNumber('General.Verbosity', verbosity)

    gmsh.model.add(meshpath.stem)  # "smallSquare8-vecEl-substractive"

    # preparing parameters
    lc = 3 * ex     # mesh size

    # defining a namespace
    gmo = gmsh.model.occ  # gmo ~ gmsh.model.opencascade
    gmg = gmsh.model.geo  # gmg ~ gmsh.model.geo

    # defining the points
    gmo.addPoint(0, 0, 0, lc, 1)
    gmo.addPoint(0, size, 0, lc, 2)
    gmo.addPoint(size, size, 0, lc, 3)
    gmo.addPoint(size, 0, 0, lc, 4)

    # adding lines
    gmo.addLine(1, 2, 1)
    gmo.addLine(2, 3, 2)
    gmo.addLine(3, 4, 3)
    gmo.addLine(4, 1, 4)

    # defining a new surface:
    gmo.addCurveLoop([1, 2, 3, 4], 20)
    surf = gmo.addPlaneSurface([20])  # full surface
    gmo.synchronize()

    # meshing the transfinite surface
    gmsh.model.mesh.setTransfiniteCurve(1, num_el_x + 1)
    gmsh.model.mesh.setTransfiniteCurve(2, num_el_x + 1)
    gmsh.model.mesh.setTransfiniteCurve(3, num_el_x + 1)
    gmsh.model.mesh.setTransfiniteCurve(4, num_el_x + 1)
    gmo.synchronize()
    gmsh.model.mesh.setTransfiniteSurface(surf, arrangement="Alternate", cornerTags=[1,2,3,4])


    # subtracting individual infill elements
    infill_points = []   # list of point tags inside the infill
    for x, y in infill_coords:
        infill_points.append(gmo.addPoint(x, y, 0, lc))  # add each point

    infill_lines = []   # list of line tags inside the infill
    infill_curves = []  # list of curve tags inside the infill
    infill_surfs = []   # list of surface tags
    for i, tri in enumerate(infill_connect):   # get each point in each triangle
        if infill_vec[i] == 0:                 # only if is to be deleted (0: not included)
            p1, p2, p3 = tri[0], tri[1], tri[2]
            tp1 = infill_points[p1]  # tag of p1
            tp2 = infill_points[p2]  # tag of p2
            tp3 = infill_points[p3]  # tag of p3
            l1 = gmo.addLine(tp1, tp2)  # tag of line 1
            l2 = gmo.addLine(tp2, tp3)  # tag of line 2
            l3 = gmo.addLine(tp3, tp1)  # tag of line 3
            xp1, yp1 = infill_coords[p1, 0], infill_coords[p1, 1]  # getting coords of p1
            xp2, yp2 = infill_coords[p2, 0], infill_coords[p2, 1]  # getting coords of p2
            xp3, yp3 = infill_coords[p3, 0], infill_coords[p3, 1]  # getting coords of p3
            infill_lines.append(l1)
            infill_lines.append(l2)
            infill_lines.append(l3)
            c1 = gmo.addCurveLoop([l1, l2, l3])
            s1 = gmo.addPlaneSurface([c1])  # create the surface
            infill_curves.append(c1)
            infill_surfs.append(s1)
    gmo.synchronize()

    surfs_to_cut = [(2, tag) for tag in infill_surfs]  # list of all surfs
    if len(surfs_to_cut) > 0:
        # substract all in one
        outlab = gmo.cut([(2, surf)], surfs_to_cut, removeObject=True, removeTool=True)
        surf = outlab[0][0][1]
        gmo.synchronize()

    # extract the curve boundary
    Boundary = [tg for tp, tg in gmsh.model.getBoundary([(2, surf)])]

    # adding physical groups
    gmsh.model.addPhysicalGroup(1, Boundary, name="SurfaceLoop") # curves physical group
    gmsh.model.addPhysicalGroup(2, [surf], name="PerimeterSurface")   # surfaces physical group
    gmo.synchronize()

    # global meshing options
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay for 2D meshes


    # generate the mesh
    gmo.synchronize()
    gmsh.model.mesh.generate(2)

    # saving the mesh
    gmsh.write(str(meshpath))

    # Launch the GUI
    if run_GUI:
        gmo.synchronize()
        gmsh.fltk.run()

    # finish the GMSH session
    gmsh.finalize()


def generate_img_file(meshpath, infill_vec, infill_connect, infill_coords, verbosity=1):
    size = 10  # physical dimension of the side
    P = 1      # number of perimeters
    ex = 0.5   # extrusion width (one perimeter width)
    width, height = size, size       # setting up the square
    num_el_x, num_el_y = 2 * size, 2 * size  # adjust the num of elements
    num_el_x_inf, num_el_y_inf = 7, 7  # adjust num of elems for infill
    dx, dy = width / num_el_x, height / num_el_y

    # initialize Gmsh
    if not gmsh.is_initialized():
        gmsh.initialize()
        gmsh.option.setNumber('General.Verbosity', verbosity)

    gmsh.model.add(meshpath.stem)  # "smallSquare8-vecEl-substractive"

    # preparing parameters
    lc = 3 * ex     # mesh size

    # defining a namespace
    gmo = gmsh.model.occ  # gmo ~ gmsh.model.opencascade
    gmg = gmsh.model.geo  # gmg ~ gmsh.model.geo

    # defining the points
    gmo.addPoint(0, 0, 0, lc, 1)
    gmo.addPoint(0, size, 0, lc, 2)
    gmo.addPoint(size, size, 0, lc, 3)
    gmo.addPoint(size, 0, 0, lc, 4)

    # adding lines
    gmo.addLine(1, 2, 1)
    gmo.addLine(2, 3, 2)
    gmo.addLine(3, 4, 3)
    gmo.addLine(4, 1, 4)

    # defining a new surface:
    gmo.addCurveLoop([1, 2, 3, 4], 20)
    surf = gmo.addPlaneSurface([20])  # full surface
    gmo.synchronize()

    # meshing the transfinite surface
    gmsh.model.mesh.setTransfiniteCurve(1, num_el_x + 1)
    gmsh.model.mesh.setTransfiniteCurve(2, num_el_x + 1)
    gmsh.model.mesh.setTransfiniteCurve(3, num_el_x + 1)
    gmsh.model.mesh.setTransfiniteCurve(4, num_el_x + 1)
    gmo.synchronize()
    gmsh.model.mesh.setTransfiniteSurface(surf, arrangement="Alternate", cornerTags=[1,2,3,4])


    # subtracting individual infill elements
    infill_points = []   # list of point tags inside the infill
    for x, y in infill_coords:
        infill_points.append(gmo.addPoint(x, y, 0, lc))  # add each point

    infill_lines = []   # list of line tags inside the infill
    infill_curves = []  # list of curve tags inside the infill
    infill_surfs = []   # list of surface tags
    for i, tri in enumerate(infill_connect):   # get each point in each triangle
        if infill_vec[i] == 0:                 # only if is to be deleted (0: not included)
            p1, p2, p3 = tri[0], tri[1], tri[2]
            tp1 = infill_points[p1]  # tag of p1
            tp2 = infill_points[p2]  # tag of p2
            tp3 = infill_points[p3]  # tag of p3
            l1 = gmo.addLine(tp1, tp2)  # tag of line 1
            l2 = gmo.addLine(tp2, tp3)  # tag of line 2
            l3 = gmo.addLine(tp3, tp1)  # tag of line 3
            xp1, yp1 = infill_coords[p1, 0], infill_coords[p1, 1]  # getting coords of p1
            xp2, yp2 = infill_coords[p2, 0], infill_coords[p2, 1]  # getting coords of p2
            xp3, yp3 = infill_coords[p3, 0], infill_coords[p3, 1]  # getting coords of p3
            infill_lines.append(l1)
            infill_lines.append(l2)
            infill_lines.append(l3)
            c1 = gmo.addCurveLoop([l1, l2, l3])
            s1 = gmo.addPlaneSurface([c1])  # create the surface
            infill_curves.append(c1)
            infill_surfs.append(s1)
    gmo.synchronize()

    surfs_to_cut = [(2, tag) for tag in infill_surfs]  # list of all surfs
    if len(surfs_to_cut) > 0:
        # substract all in one
        outlab = gmo.cut([(2, surf)], surfs_to_cut, removeObject=True, removeTool=True)
        surf = outlab[0][0][1]
        gmo.synchronize()

    # extract the curve boundary
    Boundary = [tg for tp, tg in gmsh.model.getBoundary([(2, surf)])]

    # adding physical groups
    gmsh.model.addPhysicalGroup(1, Boundary, name="SurfaceLoop") # curves physical group
    gmsh.model.addPhysicalGroup(2, [surf], name="PerimeterSurface")   # surfaces physical group
    gmo.synchronize()

    # global meshing options
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay for 2D meshes

    # setting up the appearence
    gmsh.option.setNumber("Mesh.ColorCarousel", 0)  # visibility setup
    gmsh.option.setNumber("Mesh.Triangles", 1)      # visibility setup
    gmsh.option.setNumber("Geometry.Points", 0)     # visibility setup
    gmsh.option.setNumber("Mesh.SurfaceFaces", 1)   # visibility setup
    gmsh.option.setNumber("Mesh.Lines", 0)          # visibility setup
    gmsh.option.setNumber("General.SmallAxes", 0)   # visibility setup
    gmsh.option.setNumber("Mesh.LineWidth", 2)      # width setup
    gmsh.option.setNumber("Mesh.Light", 0)          # light setup

    #gmsh.option.setColor("Geometry.Color.Points", 255, 165, 0)  # setting the color
    #gmsh.option.setColor("General.Color.Text", 255, 255, 255)   # setting the color

    gmsh.option.setColor("Mesh.Color.Lines", 210, 200, 185)       # setting the color
    gmsh.option.setColor("Mesh.Color.Triangles", 30, 30, 190)     # setting the color
    #gmsh.option.setColor("Mesh.SurfaceFaces", 0, 255, 0, 1)
    # gmsh.option.setColor("Mesh.SurfaceFaces", 0, 200, 0, 10)

    # generate the mesh
    gmo.synchronize()
    gmsh.model.mesh.generate(2)

    # Launch the GUI and saving the mesh
    gmo.synchronize()
    gmsh.fltk.initialize()
    gmsh.write(str(meshpath))

    # close the GUI
    gmsh.fltk.finalize()

    # finish the GMSH session
    gmsh.finalize()

# load solutions from the txt file
def load_txt_solutions(path_solutions):  # out_dir / "sq_solutions.txt"
    with open(path_solutions, "r") as f:
        lines = f.readlines()
        header = lines[0]
        text = lines[1:]

        X = []
        F1, F2 = [], []
        for line in text:
            sline = line.split('\t')
            vec = sline[0].replace('[','').replace(']','').replace(' ','')
            x = [int(el) for el in vec.split(',')]
            x = np.array(x)

            f1 = sline[1].replace(' ','').split(':')[1]
            f1 = float(f1)
            f2 = sline[2].replace(' ','').split(':')[1]
            f2 = float(f2)

            X.append(x)
            F1.append(f1)
            F2.append(f2)
    F1, F2 = np.array(F1), np.array(F2)
    F = np.vstack((F1, F2)).T
    X = np.vstack(X)
    return F, X
