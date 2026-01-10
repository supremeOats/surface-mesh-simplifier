import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import ArrayLike
import trimesh
import time

def edge_collapse(edge, mesh) -> None:
    vertices = mesh.vertices
    faces = mesh.faces
    v_keep, v_kill = edge   # note: v_keep and v_kill are INDICES, not vertices

    # new pos
    new_coor = (vertices[v_keep] + vertices[v_kill]) * 0.5
    vertices[v_keep] = new_coor

    # remove/replace v_kill
    vert_to_keep = np.arange(len(vertices)) != v_kill

    faces[faces == v_kill] = v_keep
    vertices[v_kill] = np.nan

    # degenerate_faces = [f for f in faces if len(np.unique(f, axis=0)) == 3]

    # update mesh
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_vertices(vert_to_keep)

def new_pos(v1, v2, Q1, Q2) -> np.ndarray:
    try:
        Q_inv = np.linalg.inv(Q1 + Q2)
    except LinAlgError:
        return np.amin([v1, v2, (v1+v2)*0.5])
    else:
        return Q_inv @ np.array([0, 0, 0, 1]).T

def adjacent_faces(vert_index, mesh):
    return mesh.faces[(mesh.faces == vert_index).all(axis=1)]

def quadratic_error(pair, Q1, Q2):    
    v1 = np.append(pair[0], 1)
    v2 = np.append(pair[1], 1)
    
    v_hat = new_pos(v1, v2, Q1, Q2)

    return v_hat.T @ (Q1 + Q2) @ v_hat  # v'(Q1 + Q2)v
    
def q_matrix(vertex, faces):
    return np.sum([
        np.power(p.T @ vertex, 2)
        for p in faces
    ])

def simplify(mesh):
    q_matrices = np.zeros_like(mesh.vertices, dtype=np.ndarray)

    for i in range(len(mesh.vertices)):
        vert_faces = adjacent_faces(i, mesh)
        q_matrices[i] = q_matrix(mesh.vertices[i], vert_faces)
    
    pair_costs = np.ndarray(2)

    for edge in mesh.edges:     # edge -> indices, pair -> positions
        pair = (mesh.vertices[edge[0]], mesh.vertices[edge[1]])
        Q1 = q_matrix(pair[0], adjacent_faces(edge[0], mesh))
        Q2 = q_matrix(pair[1], adjacent_faces(edge[1], mesh))
        
        heap = np.append(pair_costs, (quadratic_error(pair, Q1, Q2), edge))   # heap?

    # pair_costs = np.sort(pair_costs, axis=1)
    pair_costs = sorted(pair_costs, reverse=True, key=lambda tup: tup[0])

    for (cost, pair) in pair_costs:
        edge_collapse(pair, mesh)

    return mesh
    ...

#
mesh_file = "./data/boat_model.stl"

mesh = trimesh.load_mesh(mesh_file)

start = time.time()
mesh = simplify(mesh)
end = time.time()

mesh.export("./data/boat_modified.stl")

print(f"Time elapsed: {round(end - start, 3)}")