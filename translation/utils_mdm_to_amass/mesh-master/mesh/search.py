#!/usr/bin/env python
# encoding: utf-8

# Created by Matthew Loper on 2013-02-20.
# Copyright (c) 2013 MPI. All rights reserved.

"""
Searching and lookup of geometric entities
==========================================

"""


import numpy as np

__all__ = ['AabbTree', 'AabbNormalsTree', 'ClosestPointTree', 'CGALClosestPointTree']


class AabbTree(object):
    """Encapsulates an AABB (Axis Aligned Bounding Box) Tree"""
    def __init__(self, m):
        from . import spatialsearch
        # this shit return NULL
        self.cpp_handle = spatialsearch.aabbtree_compute(m.v.astype(np.float64).copy(order='C'), m.f.astype(np.uint32).copy(order='C'))

    def nearest(self, v_samples, nearest_part=False):
        "nearest_part tells you whether the closest point in triangle abc is in the interior (0), on an edge (ab:1,bc:2,ca:3), or a vertex (a:4,b:5,c:6)"
        from . import spatialsearch
        f_idxs, f_part, v = spatialsearch.aabbtree_nearest(self.cpp_handle, np.array(v_samples, dtype=np.float64, order='C'))
        return (f_idxs, f_part, v) if nearest_part else (f_idxs, v)

    def nearest_alongnormal(self, points, normals):
        from . import spatialsearch
        distances, f_idxs, v = spatialsearch.aabbtree_nearest_alongnormal(self.cpp_handle,
                                                                          points.astype(np.float64),
                                                                          normals.astype(np.float64))
        return (distances, f_idxs, v)

    def intersections_indices(self, q_v, q_f):
        '''
            Given a set of query vertices and faces, the function computes which intersect the mesh
            A list with the indices in q_f is returned
            @param q_v The query vertices (array of 3xN float values)
            @param q_f The query faces (array 3xF integer values)
        '''
        import spatialsearch
        return spatialsearch.aabbtree_intersections_indices(self.cpp_handle,
                                                            q_v.astype(np.float64),
                                                            q_f.astype(np.uint32))


class ClosestPointTree(object):
    """Provides nearest neighbor search for a cloud of vertices (i.e. triangles are not used)"""
    def __init__(self, m):
        from scipy.spatial import KDTree
        self.v = m.v
        self.kdtree = KDTree(self.v)

    def nearest(self, v_samples):
        (distances, indices) = zip(*[self.kdtree.query(v) for v in v_samples])
        return (indices, distances)

    def nearest_vertices(self, v_samples):
        (distances, indices) = zip(*[self.kdtree.query(v) for v in v_samples])
        return self.v[indices]


class CGALClosestPointTree(object):
    """Encapsulates an AABB (Axis Aligned Bounding Box) Tree """
    def __init__(self, m):
        from . import spatialsearch
        self.v = m.v
        n = m.v.shape[0]
        faces = np.vstack([np.array(range(n)), np.array(range(n)) + n, np.array(range(n)) + 2 * n]).T
        eps = 0.000000000001
        self.cpp_handle = spatialsearch.aabbtree_compute(np.vstack([m.v + eps * np.array([1.0, 0.0, 0.0]), m.v + eps * np.array([0.0, 1.0, 0.0]), m.v - eps * np.array([1.0, 1.0, 0.0])]).astype(np.float64).copy(order='C'), faces.astype(np.uint32).copy(order='C'))

    def nearest(self, v_samples):
        from . import spatialsearch
        f_idxs, f_part, v = spatialsearch.aabbtree_nearest(self.cpp_handle, np.array(v_samples, dtype=np.float64, order='C'))
        return (f_idxs.flatten(), (np.sum(((self.v[f_idxs.flatten()] - v_samples) ** 2.0), axis=1) ** 0.5).flatten())

    def nearest_vertices(self, v_samples):
        from . import spatialsearch
        f_idxs, f_part, v = spatialsearch.aabbtree_nearest(self.cpp_handle, np.array(v_samples, dtype=np.float64, order='C'))
        return self.v[f_idxs.flatten()]


class AabbNormalsTree(object):
    def __init__(self, m):
        # the weight of the normals cosine is proportional to the std of the vertices
        # the best point can be translated up to 2*eps because of the normals
        from . import aabb_normals
        eps = 0.1  # np.std(m.v)#0
        self.tree_handle = aabb_normals.aabbtree_n_compute(m.v, m.f.astype(np.uint32).copy(), eps)

    def nearest(self, v_samples, n_samples):
        from . import aabb_normals
        closest_tri, closest_p = aabb_normals.aabbtree_n_nearest(self.tree_handle, v_samples, n_samples)
        return (closest_tri, closest_p)
