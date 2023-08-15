# Single iteration from Gmsh meshing to FEniCS in functions for testing purposes
import meshio
import gmsh
from ufl import nabla_div

cwd = Path.cwd()
mesh_dir = cwd.joinpath("MOEA_meshes", "smallSquare7x7")  # adjust the dir here
inf_file = str(mesh_dir.stem) + '-infill.msh'
mesh_inp = mesh_dir / inf_file

# Extracting geometry from the infill mesh
conn, coords, num_el = extract_mesh_data(mesh_inp)

# Create the mesh infill based on the infill vector
#vec = [0,0,1,1,0,0,0,0,1,1,0,0,0,0,1,1,0,0]
vec = [0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,0,0,0,0,0,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,0,1,1,1,1,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        
vec = np.random.randint(2, size= 7 * 7 * 2)
meshfile = mesh_dir / 'infill_gen.msh'
generate_infill_file(meshfile, vec, conn, coords, run_GUI=False)

# Loading the created infill mesh to FEniCS
mesh, _, _, _ = extract_mesh_data_fenics(meshfile)

# perform FEA and obtain displacements and von mises stress
d_max, d_tot, d_avg, max_vm = square_FEA(mesh, verbose=True, plotting=True)

print(d_max)
print(max_vm)
