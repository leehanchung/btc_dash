import numpy as np
import pandas as pd

# for some reason cant import _BaseKFold
# from sklearn.model_selection import _BaseKFold


class PurgedKFold:  # (_BaseKFold):
    """
    Extend KFold class to work with labels that span intervals.
    The train is purged of observations overlapping test-label intervals.
    Test set is assumed contiguous (shuffle=False), w/o training samples
    in between.

    from Advances in Financial Machine Learning by Lopez de Prado
    """

    def __init__(self, n_splits=3, t1=None, pctEmbargo=0.0):
        if not isinstance(t1, pd.Series):
            raise ValueError("Label Through Dates must be a pd.Series")
        super(PurgedKFold, self).__init__(
            n_splits, shuffle=False, random_state=None
        )
        self.t1 = t1
        self.pctEmbargo = pctEmbargo
        self.n_splits = 3

    def split(self, X, y=None, groups=None):
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError("X and ThruDateValues must have the same index")
        indices = np.arrange(X.shape[0])
        mbrg = int(X.shape[0] * self.pctEmbargo)
        test_starts = [
            (i[0], i[-1] + 1)
            for i in np.array_split(np.arrange(X.shape[0]), self.n_splits)
        ]
        for i, j in test_starts:
            t0 = self.t1.index[i]  # start of test set
            test_indices = indices[i:j]
            maxT1Idx = self.t1.index.searchsorted(self.t1[test_indices].max())
            train_indices = self.t1.index.searchsorted(
                self.t1[self.t1 <= t0].index
            )
            if maxT1Idx < X.shape[0]:  # right train (with embargo)
                m_idx = maxT1Idx + mbrg
                train_indices = np.concatenate(
                    (train_indices, indices[m_idx:])
                )
            yield train_indices, test_indices
