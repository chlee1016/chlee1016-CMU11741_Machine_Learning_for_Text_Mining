import numpy as np
from scipy.sparse import csr_matrix
import time
from random import *
import conjugateGradient as cg

class Svm(object):
    def __init__(self, inputDim, cartegory = "realsim"):
        # self.w = None
        sigma = 1
        self.w = sigma * np.random.randn(inputDim)
        self.history = {}

        print("inputDim",inputDim)

        if cartegory == "realsim":
            print("realsim data")
            self.f_w_star = 669.664812
            self.lam = 7230.875
            self.epoch = 3000
            self.r_0 = 0.0007
            self.beta = 0.2
            self.batch_size = 1000

        if cartegory == "covtype":
            print("covtype data")
            self.f_w_star = 2541.664519
            self.lam = 3631.3203125
            self.epoch = 500
            self.r_0 = 0.00025
            self.beta = 0.2
            self.batch_size = 1000

    def train_sgd(self, X_train, y_train, X_val, y_val):
        start_time = time.time()
        hist_rfvd = []
        hist_norm = []
        hist_acc = []
        hist_time = []

        # learning rate annealing
        for i in range(self.epoch):
            r_t = self.r_0 / (1 + self.beta * i)
            if i % 10 == 0:
                print(i, "th iter")

            start_idx = randint(0, (X_train.shape[0]-1) - self.batch_size)

            # extract mini batch from original data
            X_mbatch = X_train[start_idx:start_idx+self.batch_size, :]
            y_mbatch = y_train[start_idx:start_idx+self.batch_size]

            # indexing
            I = (1 - y_mbatch * X_mbatch.dot(self.w)) > 0

            X_I = X_mbatch[I, :]
            y_I = y_mbatch[I]

            # w : d x 1 dim
            # dw : d x 1 dim
            # X_I : I x d dim
            # y_I : I x 1 dim
            dw = self.w + 2 * self.lam / self.batch_size * X_I.T.dot(X_I.dot(self.w)-y_I)
            self.w = self.w - r_t * dw

            hist_time.append(time.time() - start_time)

            # hinge : I x 1 dim
            hinge = 1 - y_mbatch * X_mbatch.dot(self.w)
            loss = 1/2 * np.dot(self.w.T, self.w) + self.lam/self.batch_size * np.dot(np.maximum(hinge,0).T, np.maximum(hinge,0))

            rfvd = (loss - self.f_w_star) / self.f_w_star
            norm = np.linalg.norm(dw, 2)
            acc = self.calAccuracy(X_val, y_val)

            hist_rfvd.append(rfvd)
            hist_norm.append(norm)
            hist_acc.append(acc)

            # if i % 10 == 0:
            print(i, "th iter")
            print("hist_relative_f", hist_rfvd[i])
            print("hist_norm", hist_norm[i])
            print("hist_acc", hist_acc[i])

        self.history['sgd_rfvd'] = hist_rfvd
        self.history['sgd_gnorm'] = hist_norm
        self.history['sgd_acc'] = hist_acc
        self.history['sgd_time'] = hist_time
        return self

    def train_ner(self, X_train, y_train, X_val, y_val):
        self.epoch = 20

        start_time = time.time()
        hist_rfvd = []
        hist_norm = []
        hist_acc = []
        hist_time = []

        for i in range(self.epoch):
            # if i % 10 == 0:
            I = (1 - y_train * X_train.dot(self.w)) > 0
            X_I = X_train[I,:]
            y_I = y_train[I]
            # w : d x 1 dim
            # dw : d x 1 dim
            # X_I : I x d dim
            # y_I : I x 1 dim
            dw = self.w + 2 * self.lam / X_train.shape[0] * X_I.T.dot(X_I.dot(self.w) - y_I)
            d, _ = cg.conjugateGradient(X_train, I, dw, self.lam)
            # H * d = - dw, d = - H^(-1) * dw
            self.w = self.w + d

            hist_time.append(time.time()-start_time)

            hinge = 1 - y_train * X_train.dot(self.w)
            loss = 1/2 * np.dot(self.w.T, self.w) + self.lam / X_train.shape[0] * np.dot(np.maximum(hinge.T,0), np.maximum(hinge,0))
            rfvd = (loss - self.f_w_star) / self.f_w_star
            acc = self.calAccuracy(X_val, y_val)
            norm = np.linalg.norm(dw, 2)

            hist_rfvd.append(rfvd)
            hist_norm.append(norm)
            hist_acc.append(acc)

            print(i, "th iter")
            print("hist_relative_f", hist_rfvd[i])
            print("hist_norm", hist_norm[i])
            print("hist_acc", hist_acc[i])

        self.history['ner_rfvd'] = hist_rfvd
        self.history['ner_gnorm'] = hist_norm
        self.history['ner_acc'] = hist_acc
        self.history['ner_time'] = hist_time

        return self


    def predict(self, x, ):
        yPred = np.zeros(x.shape[0])

        # x : n x d
        # w : d x 1
        s = x.dot(self.w)
        for i in range(np.size(x, axis=0)):
            # decision boundary
            if s[i] >= 0:
                yPred[i] = 1
            else:
                yPred[i] = -1
        return yPred

    def calAccuracy(self, x, y):
        acc = 0
        yPred = self.predict(x)
        acc = np.mean(y == yPred) * 100
        return acc

