import libmr

import numpy as np
# from numba import float_, int_, bool_
import sklearn.metrics.pairwise
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator
from sklearn.datasets import load_digits
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import argparse
import pickle


def euclidean_cdist(X, Y):
    return sklearn.metrics.pairwise.pairwise_distances(X, Y, metric="euclidean", n_jobs=1)


def euclidean_pdist(X):
    return sklearn.metrics.pairwise.pairwise_distances(X, metric="euclidean", n_jobs=1)


def cosine_cdist(X, Y):
    return sklearn.metrics.pairwise.pairwise_distances(X, Y, metric="cosine", n_jobs=1)


def cosine_pdist(X):
    return sklearn.metrics.pairwise.pairwise_distances(X, metric="cosine", n_jobs=1)


dist_func_lookup = {
    "cosine": {"cdist": cosine_cdist,
               "pdist": cosine_pdist},

    "euclidean": {"cdist": euclidean_cdist,
                  "pdist": euclidean_pdist}
}

ap = argparse.ArgumentParser()

ap.add_argument("--embeddings", default="outputs/embeddings.pickle", help='Path to embeddings')
ap.add_argument("--tailsize", type=int, help="number of points that constitute \'extrema\'", default=50)
ap.add_argument("--cover_threshold", type=float, help="probabilistic threshold to designate redundancy between points",
                default=1)
# ap.add_argument("--distance",type=str,default="cosine",choices=dist_func_lookup.keys())
ap.add_argument("--nfuse", type=int, help="number of extreme vectors to fuse over", default=1)
ap.add_argument("--margin_scale", type=float, help="multiplier by which to scale the margin distribution", default=0.5)
ap.add_argument("--weibull", default="outputs/weibull.pickle", help='Path save weibulls')
ap.add_argument("--distance", type=str, default="euclidean", choices=dist_func_lookup.keys())
ap.add_argument("--prob_threshold", type=float, help="probabilistic threshold to unknown or known", default=0.8)

args = ap.parse_args()

pdist_func = dist_func_lookup[args.distance]["pdist"]


class EVM(BaseEstimator):
    UNKNOWN = "Unknown"

    def __init__(self,
                 tail: int = 10,
                 open_set_threshold: float = 0.5,
                 biased_distance: float = 0.5,
                 k: int = 5,
                 redundancy_rate: float = 0,
                 use_gpu=False):
        self.tail = tail
        self.biased_distance = biased_distance
        self.classes = {}
        self.dists = {}
        self.open_set_threshold = args.prob_threshold
        self.k = k
        self.redundancy_rate = redundancy_rate
        self.use_gpu = use_gpu
        self.cdist_func = dist_func_lookup[args.distance]["cdist"]

    def fit(self, X, y):
        classes = np.unique(y)
        for clz in classes:
            self.classes[clz] = X[y == clz]
        self._infer_classes(classes)
        if self.redundancy_rate > 0:
            self._reduce()

    def remove(self, clz):
        self.classes.pop(clz)
        self.dists.pop(clz)

    def findCosineDistance(self, vector1, vector2):
        """
        Calculate cosine distance between two vector
        """
        vec1 = vector1.flatten()
        vec2 = vector2.flatten()

        a = np.dot(vec1.T, vec2)
        b = np.dot(vec1.T, vec1)
        c = np.dot(vec2.T, vec2)
        return (a / (np.sqrt(b) * np.sqrt(c)))

    def _infer(self):
        self._infer_classes(list(self.classes.keys()))

    def _infer_classes(self, indices):
        temp = [self._infer_class(i) for i in indices]
        for i in range(len(temp)):
            self.dists[indices[i]] = temp[i]

    def _infer_class(self, class_index):
        in_class = self.classes[class_index]
        out_class = np.concatenate([self.classes[i] for i in self.classes.keys() if i != class_index])
        distances = cdist(in_class, out_class, metric='cosine')
        distances.sort(axis=1)
        distances = self.biased_distance * distances[:, :self.tail]
        return np.apply_along_axis(self._fit_weibull, 1, distances)

    def _fit_weibull(self, row):
        mr = libmr.MR()
        mr.fit_low(row, self.tail)
        return mr

    def predict(self, X):
        return np.apply_along_axis(self._predict_row, 1, X)

    # def predict_with_prop(self, X):
    #    return np.apply_along_axis(self._predict_with_prob, 1, X)

    def predict_with_prob(self, row):
        max_prop, max_class = -1, -1
        for i in self.classes.keys():
            clz, dist = self.classes[i], self.dists[i]
            # distances = np.linalg.norm(clz - row, axis=1, keepdims=True)
            # distances = self.cdist_func(clz,row)
            distances = cdist(clz, row, metric='cosine')
            props = [dist[j].w_score(distances[j]) for j in range(dist.shape[0])]
            prop = max(props)
            if prop > max_prop:
                max_prop = prop
                max_class = i
        return (max_class, max_prop) if max_prop >= self.open_set_threshold else (EVM.UNKNOWN, 1 - max_prop)

    def _predict_class(self, item):
        row, class_index = item
        clz, dist = self.classes[class_index], self.dists[class_index]
        distances = np.linalg.norm(clz - row, axis=1, keepdims=True)
        props = [dist[j].w_score(distances[j]) for j in range(dist.shape[0])]
        prop = max(props)
        return prop, class_index

    def _reduce(self):
        for i in self.classes.keys():
            self._reduce_class(i)

    def _reduce_class(self, class_index):
        clz, dist = self.classes[class_index], self.dists[class_index]
        distances = cdist(clz, clz)
        l = clz.shape[0]
        s = [0] * l
        for i in range(l):
            s[i] = []
            for j in range(l):
                if dist[i].w_score(distances[i, j]) >= self.redundancy_rate:
                    s[i].append(j)
        u = set(range(l))
        c = set()
        indices = []
        s = [set(l) for l in s]
        len_s = [len(j) for j in s]
        while u != c:
            temp = max(len_s)
            ind = len_s.index(temp)
            c |= s[ind]
            indices.append(ind)
            s.pop(ind)
            len_s.pop(ind)
        self.classes[class_index], self.dists[class_index] = self.classes[class_index][indices], \
                                                             self.dists[class_index][indices]


if __name__ == '__main__':
    data = pickle.loads(open(args.embeddings, "rb").read())
    Xtrain = np.array(data['embeddings'])
    ytrain = data['names']
    ytrain = np.array(ytrain).reshape(-1)
    print(f'X shape: {Xtrain.shape}')
    best_estimator = EVM(tail=args.tailsize, open_set_threshold=0)

    # grid = GridSearchCV(estimator, param_grid=params, scoring=make_scorer(accuracy_score), n_jobs=-1)
    best_estimator.fit(Xtrain, ytrain)
    f = open(args.weibull, "wb")
    f.write(pickle.dumps(best_estimator))
    f.close()
    # predicted = best_estimator.predict(X_test)
    # accuracy = (predicted == y_test).sum() * 100 / X_test.shape[0]
    # print("best accuracy = {}".format(accuracy))
