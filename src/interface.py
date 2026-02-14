import trimesh
import time
from os import listdir, curdir
from os.path import join, isfile
from simplifier import simplify

DATA_DIR = join("./data")

def print_available_file_names() -> None:
    print("Available files:")

    for file in listdir(DATA_DIR):
        if file[-3:] == "stl":
            print(file)

def file_name_from_input() -> str:
    print("\nChoose file:")
    path = ""
    file = ""

    while not isfile(path):
        file = input()
        path = join(DATA_DIR, file)
        
        if not isfile(path):
            print("Invalid file name\nChoose file:")

    return file

def run():
    print_available_file_names()    

    file = file_name_from_input()
            
    mesh = trimesh.load_mesh(join(DATA_DIR, file))
    
    start = time.time()
    mesh = simplify(mesh, 50)
    end = time.time()

    print(f"\nTime elapsed: {round(end - start, 3)}")

    mesh.export(join(DATA_DIR, "results", f"{file[:-4]}_modified_{time.time()}.stl"))

run()
