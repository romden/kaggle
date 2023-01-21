import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, ClassifierMixin


def fit_model(x, y, kernel='linear', alpha=10):
    return KernelRidge(alpha=alpha, kernel=kernel).fit(x, y)


class CustomKernelRegressor(BaseEstimator, RegressorMixin):

    def __init__(self):        
        self.n_rounds = 5
        self.kernel = 'linear' #RBF(length_scale=scale)
        self.alpha = 10

    def fit(self, X, y, **kwargs):
        if y is None:
            raise ValueError('requires y to be passed, but the target y is None')

        self.models = []        
        n = X.shape[0]

        for _ in range(self.n_rounds):
            
            batches = np.random.permutation(n)            

            idx = batches[:n//2]
            model = fit_model(X[idx], y[idx], self.kernel, self.alpha)
            models.append(model)
            
            idx = batches[n//2:]
            model = fit_model(X[idx], y[idx], self.kernel, self.alpha)
            models.append(model)

        return self

    def predict(self, X):
        y = None
        for model in models:
            y_tmp = model.predict(X)
            if y is None:
                y = y_tmp
            else:
                y += y
        y /= len(models)
        return y