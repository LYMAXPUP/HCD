import pandas as pd
from sklearn.model_selection import train_test_split


class Dataset:
    """Storage data and split data into training and testing set: train_id and test_id.
     ratio : double in (0, 1)
        The ratio of training set in dataset.

    """
    def __init__(self, y, y_trans, ratio):
        self.y = y
        y = pd.DataFrame(y)
        X = y
        x_train, x_test, _, _ = train_test_split(X, y, train_size=ratio, random_state=None,
                                                 shuffle=True, stratify=y)
        self.train_id = x_train[0].index.tolist()
        self.test_id = x_test[0].index.tolist()
        self.y_trans = y_trans
