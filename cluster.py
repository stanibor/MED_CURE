import numpy as np

INF = float('inf')


class Cluster:
    def __init__(self, dataset, indices: np.ndarray):
        assert indices.ndim == 1, "Indices should be one-dimensional"
        self.dataset = dataset
        self.indices = indices
        self.n_points = len(indices)
        self.mean = dataset[indices].mean(0)
        self.rep = self.mean

        self.closest = None
        self.closest_dist = INF

    @property
    def points(self):
        return self.dataset[self.indices]

    def set_representatives(self, c, alpha):
        points = self.points
        diffs = points - self.mean
        self.rep = np.zeros_like(points[:c])
        if c >= self.n_points:
            self.rep[:] = diffs[0]
            self.rep[:self.n_points] = diffs[:self.n_points]
            self.rep *= alpha
            self.rep += self.mean
            return

        for i in range(c):
            index = distance_matrix(diffs, self.rep[:(i+1)]).min(-1).argmax()
            self.rep[i] = diffs[index]
        self.rep *= alpha
        self.rep += self.mean

    def merge(self, v, c: int, alpha: float):
        # assert v.dataset is self.dataset
        self.indices = np.concatenate((self.indices, v.indices))
        self.mean = (self.n_points * self.mean + v.n_points * v.mean) / (self.n_points + v.n_points)
        self.n_points += v.n_points
        self.set_representatives(c, alpha)

    def distance(self, v):
        return distance_matrix(self.rep, v.rep).min()

    def set_closest(self, closest):
        self.closest = closest
        self.closest_dist = self.distance(closest)


def distance_matrix(a_points: np.ndarray, b_points: np.ndarray):
    return (np.expand_dims(a_points, 1) - np.expand_dims(b_points, 0)).__pow__(2).sum(-1)


def representatives(clusters, c):
    out = np.zeros((len(clusters), c, clusters[0].points.shape[-1]))
    for i, cluster in enumerate(clusters):
        n_reps = cluster.rep.shape[0]
        out[i, :] = cluster.rep[0]
        out[i, :n_reps] = cluster.rep
    return out


def index_of_closest_reps(reps, i):
    r = reps[i:(i+1)]
    dists = distance_matrix(reps, r)
    dists[i] = dists.max()
    return dists.min(-1).argmin()





