import trimesh
import time
from os import listdir, curdir
from os.path import join
from simplifier import simplify

DATA_DIR = join("./data")

def run():
    print("Available files:")
    
    for file in listdir(DATA_DIR):
        if file[-3:] == "stl":
            print(file)
    
    print("\nChoose file:")
    file = input()
    
    mesh = trimesh.load_mesh(join(DATA_DIR, file))
    
    start = time.time()
    mesh = simplify(mesh, 200)
    end = time.time()

    print(f"\nTime elapsed: {round(end - start, 3)}")

    mesh.export(join(DATA_DIR, f"{file[:-4]}_modified_{time.time()}.stl"))

run()

'''
mesh_file = "./data/cube_triangulated.stl"

mesh = trimesh.load_mesh(mesh_file)

start = time.time()
mesh = simplify(mesh)
end = time.time()

mesh.export("./data/cube_modified.stl")

print(f"Time elapsed: {round(end - start, 3)}")
'''