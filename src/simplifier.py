import heapq
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
    
    try:
        Q_inv = np.linalg.inv(Q1 + Q2)
    except LinAlgError:
        #todo
        return (v1+v2)*0.5
    
    else:
        return Q_inv @ np.array([0, 0, 0, 1]).T

def adjacent_faces(vert_index, mesh):
    return mesh.faces[vert_index]

def quadratic_error(pair, Q1, Q2):
    v1 = np.append(pair[0], 1)
    v2 = np.append(pair[1], 1)
    
    v_hat = new_pos(v1, v2, Q1, Q2)

    return v_hat.T @ (Q1 + Q2) @ v_hat  # v'(Q1 + Q2)v

def k_matrix(face, vertices):
    v1, v2, v3 = vertices[face]
    norm_vec = np.cross(v2-v1, v3-v1)       # a,b,c
    # norm_vec /= np.linalg.norm(norm_vec)    # normalize
    #todo

    p = np.append(norm_vec, -np.dot(norm_vec, v1))
    # ^ a.x1 + b.x1 + c.x1 + d = 0 =>
    # d = -(a.x1 + b.x1 + c.x1) = -n.v1

    return np.outer(p, p.T)
    
def q_matrix(vertex, faces, mesh):
    Q = np.zeros((4,4))
    for f in faces:
        f = mesh.faces[f]
        Q += k_matrix(f, mesh.vertices)

    return Q

def simplify(mesh, iterations=0):
    # pair_costs = np.ndarray(2)
    heap = []

    for edge in mesh.edges:     # edge -> indices, pair -> positions
        pair = (mesh.vertices[edge[0]], mesh.vertices[edge[1]])
        Q1 = q_matrix(pair[0], adjacent_faces(edge[0], mesh), mesh)
        Q2 = q_matrix(pair[1], adjacent_faces(edge[1], mesh), mesh)
        
        err = quadratic_error(pair, Q1, Q2)
        
        heapq.heappush(heap, (err, tuple(edge)))   # when tuple given heapq sorts by the first value
    
    for i in range(iterations):
        edge_collapse(heap[i][1], mesh)

    return mesh

