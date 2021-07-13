import numpy as np
import networkx as nx
from numpy.linalg import norm
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise

from evaluateSVM import EvaluateSVM


def get_proximity_similarity_matrix(G):
    Aij = np.array(nx.adjacency_matrix(G).todense())
    S = (Aij + pairwise.cosine_similarity(Aij)) / 2
    S = normalize(S, axis=1, norm='l2')
    return S


class LearnBestX:
    """The process of learning the best matrix X after many steps of iteration.
    The iteration: X -> SVM -> W,z -> X -> Y -> X
        Initialize X and Y;
        loop:
            input X to train the SVM, return parameters W,z
            use W,z to adjust X
            use X to update Y, then use Y to update X.
    """
    C = 1
    eta = 0.001
    lmbda = 1

    def __init__(self, G3_i, low_dim, data, Central):
        self.G = G3_i
        np.random.seed(0)
        self.X = np.array(np.random.rand(len(G3_i.nodes), low_dim), dtype=np.float32)
        self.X = normalize(self.X, axis=0, norm='l2')
        self.H = normalize(Central, axis=1, norm='l2')
        self.Y = np.array(np.random.rand(low_dim, self.H.shape[0]), dtype=np.float32)
        self.bias = np.zeros(self.X.shape, dtype=np.float32)
        self.num_role_kinds = data.y_trans.shape[1]
        self.weight_b = np.zeros((self.num_role_kinds, low_dim))  # W
        self.alpha_b = np.zeros((1, self.num_role_kinds))  # z
        self.data = data

    def pre_compute_bias(self):
        # (num_role_kinds, low_dim)
        W = self.weight_b
        # (1, num_role_kinds)
        z = self.alpha_b.flatten()
        c = np.zeros((1, self.num_role_kinds))
        for id in self.data.train_id:
            c = self.data.y_trans[id] * self.C
            labels = np.where(self.data.y_trans[id] == True)[0]
            for label in labels:
                for j in range(self.num_role_kinds):
                    self.bias[id] += (c[j] - z[j]) * (W[label] - W[j])

    def train_w(self):
        last_loss = 0.0
        M = get_proximity_similarity_matrix(self.G)

        # Adjust X according to SVM result
        self.pre_compute_bias()
        self.X += self.eta * self.bias

        # Update Y
        np.random.seed(0)
        self.Y = np.array(np.random.rand(self.Y.shape[0], self.Y.shape[1]))
        self.Y = normalize(self.Y, axis=1, norm='l2')
        for step in range(10):
            # L对Y求导
            drv = np.zeros(self.Y.shape, dtype=np.float32)
            drv += 2 * self.lmbda * self.Y - 2 * np.dot(np.dot(self.X.T, M), self.H.T) \
                   + 2 * np.dot(np.dot(np.dot(np.dot(self.X.T, self.X), self.Y), self.H), self.H.T)
            rt = ((- drv.T).flatten()).reshape((self.Y.shape[0] * self.Y.shape[1], 1))
            dt = rt
            drv.fill(0.0)
            vecY = (self.Y.T).reshape((self.Y.shape[0] * self.Y.shape[1], 1))
            countY = 0
            while True:
                countY += 1
                if countY > 10:
                    break
                else:
                    dtM = dt.reshape((self.Y.shape[1], self.Y.shape[0])).T
                    storeYdt = 2 * np.dot(np.dot(np.dot(np.dot(self.X.T, self.X), dtM), self.H), self.H.T)\
                               + 2 * self.lmbda * dtM
                    Ydt = ((storeYdt.T).flatten()).reshape((self.Y.shape[0] * self.Y.shape[1], 1))
                    rtrt = norm(rt) ** 2
                    dtHdt = np.sum(np.multiply(dt, Ydt))
                    # Learning rate
                    at = rtrt / dtHdt
                    vecY = vecY + at * dt
                    rt = rt - at * Ydt
                    rtmprtmp = rtrt
                    rtrt += norm(rt) ** 2
                    bt = rtrt / rtmprtmp
                    dt = rt + bt * dt
            Yt = (vecY.flatten()).reshape((self.Y.shape[1], self.Y.shape[0]))
            self.Y = Yt.T

            XYH = np.dot(np.dot(self.X, self.Y), self.H)
            fitMXYH = norm(M - XYH) ** 2
            loss = fitMXYH + (self.lmbda) * (norm(self.X) ** 2 + norm(self.Y) ** 2)
            if abs(last_loss - loss) < 5e-11:
                break
            last_loss = loss

        # Update X
        np.random.seed(0)
        self.X = np.array(np.random.rand(self.X.shape[0], self.X.shape[1]))
        self.X = normalize(self.X, axis=0, norm='l2')
        for step in range(10):
            # L对X求导
            drv = np.zeros(self.X.shape, dtype=np.float32)
            drv += 2 * self.lmbda * self.X - 2 * np.dot(np.dot(M, self.H.T), self.Y.T) \
                   + 2 * np.dot(np.dot(np.dot(np.dot(self.X, self.Y), self.H), self.H.T), self.Y.T)
            rt = ((- drv.T).flatten()).reshape((self.X.shape[0] * self.X.shape[1], 1))
            dt = rt
            drv.fill(0.0)
            vecX = (self.X.T).reshape((self.X.shape[0] * self.X.shape[1], 1))
            countE = 0
            while True:
                countE += 1
                if countE > 10:
                    break
                else:
                    dtM = dt.reshape((self.X.shape[1], self.X.shape[0])).T
                    storeXdt = 2 * np.dot(np.dot(np.dot(np.dot(dtM, self.Y), self.H), self.H.T), self.Y.T) + 2 * self.lmbda * dtM
                    Xdt = ((storeXdt.T).flatten()).reshape((self.X.shape[0] * self.X.shape[1], 1))
                    rtrt = norm(rt) ** 2
                    dtXdt = np.sum(np.multiply(dt, Xdt))
                    at = rtrt / dtXdt
                    vecX = vecX + at * dt
                    rt = rt - at * Xdt
                    rtmprtmp = rtrt
                    rtrt += norm(rt) ** 2
                    bt = rtrt / rtmprtmp
                    dt = rt + bt * dt
            Xt = (vecX.flatten()).reshape((self.X.shape[1], self.X.shape[0]))
            self.X = Xt.T

            XYH = np.dot(np.dot(self.X, self.Y), self.H)
            fitMXYH = norm(M - XYH) ** 2
            loss = fitMXYH + self.lmbda * (norm(self.X) ** 2 + norm(self.Y) ** 2)
            if abs(last_loss - loss) < 5e-11:
                break
            last_loss = loss
        return self.X

    def run(self):
        X = np.zeros(self.X.shape)
        bestX = np.zeros(self.X.shape)
        f1_micro = 0.0
        # 10 iterations
        for i in range(10):
            if i == 0:
                X = self.train_w()
                bestX = X
            else:
                svm = EvaluateSVM(self.data, X, self.data.y, self.data.y_trans).predict()
                self.C = svm[2]
                self.weight_b, self.alpha_b = svm[0], svm[1]
                new_f1_micro = svm[3]
                X = self.train_w()
                if new_f1_micro > f1_micro:
                    bestX = X
        return bestX




