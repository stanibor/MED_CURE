import numpy as np
from typing import List
import matplotlib.pyplot as plt
from cluster import Cluster, INF, distance_matrix, representatives, index_of_closest_reps
from tqdm import tqdm



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



if __name__ == '__main__':
    a = np.random.randn(5000, 12)
    w = Cluster(a, np.arange(len(a)))
    w.set_representatives(4, 0.8)

    cluster_list = make_cluster_list(a)
    cluster_list = cluster(cluster_list, 10)
    print(len(cluster_list))

    for c in cluster_list:
        plt.scatter(c.points[:, 0], c.points[:, 1])
        plt.scatter(c.rep[:, 0], c.rep[:, 1], marker='+')
    plt.show()

