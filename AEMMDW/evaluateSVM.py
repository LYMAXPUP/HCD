import sklearn.svm as svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from warnings import filterwarnings


class EvaluateSVM:
    """Input the matrix X to train the SVM model (adjust the hyperparameter C).
     And the parameters `weight_b` and `alpha_b` returned by SVM will assist to update
     matrices X and Y in `learnBestX.py`.

    """
    # 返回W、Z、C
    C = 10

    def __init__(self, dataset, X, y, y_trans):
        self.data = dataset
        self.seed = np.random.randint(10)
        self.X = normalize(X, axis=0, norm='l2')
        self.y = y
        self.y_trans = y_trans

    def train(self):
        error = []
        # Can adjust to larger iterations.
        e_range = list(range(1, 200))
        for i in e_range:
            clf = svm.SVC(C=i,
                          kernel='linear',
                          decision_function_shape='ovr',
                          random_state=self.seed,
                          probability=True,
                          class_weight='balanced')
            scores = cross_val_score(estimator=clf,
                                     X=self.X[self.data.train_id, :],
                                     y=self.y[self.data.train_id],
                                     cv=2,
                                     scoring='neg_log_loss')
            error.append(1 - scores.mean())
        k = error.index(min(error))
        print("the best C=", e_range[k])
        # Ignore the ConvergenceWarning
        filterwarnings("ignore")
        self.C = e_range[k]
        clf_n = OneVsRestClassifier(LinearSVC(C=self.C, random_state=self.seed, class_weight='balanced'))
        clf_n.fit(self.X[self.data.train_id, :], self.y_trans[self.data.train_id, :])
        return clf_n

    def predict(self):
        clf = self.train()
        # w矩阵(num_role_kinds, low_dim)
        weight_b = clf.coef_
        # w0向量(1, num_role_kinds)
        alpha_b = clf.intercept_.flatten()
        y_pred = clf.predict(self.X[self.data.test_id, :])
        f1_micro = f1_score(self.y_trans[self.data.test_id, :], y_pred, average='micro')
        f1_macro = f1_score(self.y_trans[self.data.test_id, :], y_pred, average='macro')
        accuracy = accuracy_score(self.y_trans[self.data.test_id, :], y_pred,)
        return (weight_b, alpha_b, self.C, f1_micro, f1_macro, accuracy, y_pred)



