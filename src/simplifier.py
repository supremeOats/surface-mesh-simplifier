import heapq
import os
import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import ArrayLike

def edge_collapse(edge, mesh) -> None:
    """
    :param edge: tuple of two indices of vertecies in the mesh
    :param mesh: mesh object
    """

    vertices = mesh.vertices
    faces = mesh.faces
    v_keep, v_kill = edge   # note: v_keep and v_kill are INDICES, not vertices

    # print(f"v_keep: {vertices[v_keep]}")
    # print(f"v_kill: {vertices[v_kill]}")

    if (v_keep >= len(vertices) or v_kill >= len(vertices)
            or any(vertices[v_keep]) == np.nan or any(vertices[v_kill]) == np.nan):
        return None

    # new pos
    new_coor = (vertices[v_keep] + vertices[v_kill]) * 0.5
    vertices[v_keep] = new_coor

    # remove/replace v_kill
    vert_to_keep = np.arange(len(vertices)) != v_kill

    faces[faces == v_kill] = v_keep
    vertices[v_kill] = np.nan

    # update mesh
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_vertices(vert_to_keep)

def new_pos(v1, v2, Q1, Q2) -> np.ndarray:
    """
    Docstring for new_pos
    
    :param v1: tuple of three floats
    :param v2: tuple of three floats
    :param Q1: Description
    :param Q2: Description
    :return: Description
    :rtype: ndarray[_AnyShape, dtype[Any]]
    """
    
    # try:
    #     Q = Q1 + Q2
    #     Q_inv = np.linalg.inv(Q)
    # except LinAlgError:
    #     #todo
    #     return (v1+v2)*0.5
    
    # else:
    #     return np.dot(Q_inv, np.array([0, 0, 0, 1]).T)

    return (v1+v2)/2

def adjacent_faces(vert_index, mesh):
    return mesh.faces[vert_index]

def quadratic_error(pair, Q1, Q2):
    v1 = np.append(pair[0], 1)
    v2 = np.append(pair[1], 1)
    
    v_hat = new_pos(v1, v2, Q1, Q2)

    Q = Q1 + Q2

    return np.dot(np.dot(   # v'(Q1 + Q2)v
        v_hat.T, Q),
        v_hat
    ) 

def k_matrix(face, vertices):
    v1, v2, v3 = vertices[face]
    norm_vec = np.cross(v2-v1, v3-v1)       # a,b,c
    norm_vec /= np.linalg.norm(norm_vec)    # normalize

    p = np.append(norm_vec, -np.dot(norm_vec, v1))
    # # ^ a.x1 + b.y1 + c.z1 + d = 0 =>
    # # d = -(a.x1 + b.x1 + c.x1) = -n.v1

    return np.outer(p, p.T)
    
def q_matrix(vert_idx, mesh):
    faces = adjacent_faces(vert_idx, mesh)

    Q = np.zeros((4,4))
    for f in faces:
        f = mesh.faces[f]
        Q += k_matrix(f, mesh.vertices)

    return Q

def edges_set(edges) -> np.ndarray:
    ...

def update_heap(heap, edge, mesh):
    kept, killed = edge
    
    for entry in heap:
        (err, (v1, v2)) = entry
        
        if v1 == killed or v2 == killed:
            if v1 == killed and v2 == kept:
                heap.remove(entry)
            elif v1 == killed:
                v1 = kept
            else:
                v2 = kept

            Q1 = q_matrix(edge[0], mesh)
            Q2 = q_matrix(edge[1], mesh)
        
            # todo - goes out of bounds; debug
            pair = (mesh.vertices[v1], mesh.vertices[v2])
                                            # ^ index 330 is out of bounds for axis 0 with size 329
        
            err = quadratic_error(pair, Q1, Q2)

def simplify(mesh, target_size=100):
    """
    :param mesh: trimesh.base.Trimesh
    :param target_size: percentage of the original size, 0-100
    """

    edges = edges_set(mesh.edges)

    heap = []

    print("Calculating error...")

    for edge in mesh.edges:     # edge -> indices, pair -> positions
        Q1 = q_matrix(edge[0], mesh)
        Q2 = q_matrix(edge[1], mesh)
        
        pair = (mesh.vertices[edge[0]], mesh.vertices[edge[1]])
        
        err = quadratic_error(pair, Q1, Q2)
        heapq.heappush(heap, (err, tuple(edge)))   # when tuple given heapq sorts by the first value
        
    print("Heap built")

    total_size = len(heap)
    curr_size = total_size

    print("Simplifing...")

    while heap and curr_size > total_size * target_size/100:
        (_, curr_edge) = heap.pop()
        edge_collapse(curr_edge, mesh)
        update_heap(heap, curr_edge, mesh)

        curr_size = len(heap)
        percentage_done = round(curr_size / total_size * 100, 1)
        if percentage_done % 10 == 2: print(f"{percentage_done}%...")

    return mesh

