import numpy as np
from typing import List
from cluster import Cluster, INF, distance_matrix, representatives, index_of_closest_reps

def make_cluster_list(points: np.ndarray):
    clusters = [Cluster(points, np.array([i])) for i in range(len(points))]

    for i, clust in enumerate(clusters):
        distances = distance_matrix(points, points[i:(i+1)])
        distances[i] = distances.max()
        closest = distances.argmin()
        clust.set_closest(clusters[closest])
    return clusters


def cluster(cluster_list: List[Cluster], k, c = 4, alpha = 0.8):
    # cluster_list = sorted(cluster_list, key=lambda x: x.closest_dist)
    for cluster in cluster_list:
        cluster.set_representatives(c, alpha)
    for iter in tqdm(np.arange(len(cluster_list) - k)):
        cluster_list = sorted(cluster_list, key=lambda x: x.closest_dist)
        w = cluster_list[0]
        u = w.closest
        w.merge(u, c, alpha)
        cluster_list.remove(u)
        # cluster_list.remove(w)
        w.closest = None
        w.closest_dist = INF
        reps =  representatives(cluster_list, c)
        for i, q in enumerate(cluster_list):
            if q is w:
                continue
            dist = w.distance(q)
            if (q.closest is u) or (q.closest is w):
                # print(id(u))
                new_closest = cluster_list[index_of_closest_reps(reps, i)]
                q.set_closest(new_closest)
            if dist < w.closest_dist:
                w.closest = q
                w.closest_dist = dist
            if dist < q.closest_dist:
                q.closest = w
                q.closest_dist = dist
    return cluster_list