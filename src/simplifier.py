import heapq
import os
import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import ArrayLike

def edge_collapse(edge, quadric, mesh) -> int:
    """
    :param edge: tuple of two indices of vertecies in the mesh
    :param mesh: mesh object
    """

    vertices = mesh.vertices
    faces = mesh.faces
    v_keep, v_kill = edge   # note: v_keep and v_kill are INDICES, not vertices

    # new pos
    Q = quadric[v_keep] + quadric[v_kill]
    mesh.vertices[v_keep] = new_pos(v_keep, v_kill, Q, vertices)[:3]
    quadric[v_keep] += quadric[v_kill]

    # remove/replace v_kill
    mesh.faces[faces == v_kill] = v_keep
    vertices[v_kill] = np.nan

    # update mesh
    mesh.update_faces(mesh.nondegenerate_faces())

    return v_keep

def new_pos(v1, v2, Q, vertices) -> np.ndarray:
    res = np.array(0.5 * (vertices[v1] + vertices[v2]))
    return np.append(res, 1)

    """
    Calculate optimal position for new vertex (minimal value of: v_TQv)
    Q =[[q11 , q12, q13, b1],
        [q12 , q22, q23, b2],
        [q13 , q23, q33, b3],
        [b1  ,  b2,  b3,  c]]
        
    Q =[[A,  b],
        [bT, c]]

    [x, y, z, 1] @ Q @ [x, y, z, 1]^T
    p = [x, y, z]
    
    A*p = -b
    """

    A = Q[:3, :3]
    b = Q[:3, 3]

    candidates = []

    try:
        p = np.linalg.solve(A, -b)
        candidates.append(p)
    except np.linalg.LinAlgError:
        pass

    pu = vertices[v1]
    pv = vertices[v2]
    pm = 0.5 * (pu + pv)

    candidates.extend([pu, pv, pm])

    def error(p):
        ph = np.append(p, 1.0)
        return ph.T @ Q @ ph

    best = min(candidates, key=error)
    return np.append(best, 1.0)
    
def adjacent_faces(vert_index, mesh):
    return mesh.faces[vert_index]

def quadric_error(edge, Q, vertices):
    (u, v) = edge
    v_opt = new_pos(u, v, Q, vertices)  # returns [x,y,z,1]

    return np.dot(np.dot(v_opt.T, Q), v_opt)
    ...
    
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

def valid_edge(edge, mesh) -> bool:
    v0, v1 = edge

    if v0 == v1:
        return False

    n = len(mesh.vertices)
    if not (0 <= v0 < n and 0 <= v1 < n):
        return False

    if np.any(np.isnan(mesh.vertices[v0])) or np.any(np.isnan(mesh.vertices[v1])):
        return False

    if v1 not in mesh.vertex_neighbors[v0]:
        return False

    return True

def vertices_error(mesh):
    quadric = []

    for idx in range(len(mesh.vertices)):
        quadric.append(q_matrix(idx, mesh))

    return quadric

def update_neighbors(kept, neighbors, heap, quadrics, vertices):
    for v in neighbors:
        v1, v2 = min(v, kept), max(v, kept)
        Q = quadrics[v1] + quadrics[v2]

        cost = quadric_error((v1, v2), Q, vertices)

        heapq.heappush(heap, (cost, (v1, v2)))

def build_heap(quadric, mesh) -> list[tuple[float, tuple]]:
    heap = []

    for (u, v) in mesh.edges:
        u, v = min(u, v), max(u, v)

        Q = quadric[u] + quadric[v]
        err = quadric_error((u, v), Q, mesh.vertices)

        heapq.heappush(heap, (err, (u, v)))
        
    return heap

def contract_edges(heap, quadrics, mesh, target):
    total_size = len(mesh.vertices)
    alive_verts = total_size

    while heap and alive_verts > total_size * target/100:
        (err, curr_edge) = heapq.heappop(heap)

        if not valid_edge(curr_edge, mesh):
            continue

        kept = edge_collapse(curr_edge, quadrics, mesh)
        update_neighbors(kept, mesh.vertex_neighbors[kept], heap, quadrics, mesh.vertices)
        
        alive_verts -= 1
        
    print(f"Curr: {alive_verts}\nTotal: {total_size}\nReduced: {total_size * target/100}")

def simplify(mesh, target_size=100):
    """
    :param mesh: trimesh.base.Trimesh
    :param target_size: percentage of the original size, 0-100
    """

    print("Calculating error...")
    weighted_vertices = vertices_error(mesh)
    
    print(f"Building heap...")
    heap = build_heap(weighted_vertices, mesh)

    print("Contracting edges...")
    contract_edges(heap, weighted_vertices, mesh, target_size)

    return mesh

