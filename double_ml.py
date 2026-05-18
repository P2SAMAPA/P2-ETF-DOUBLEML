import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
import warnings

try:
    from econml.dml import LinearDML
    HAS_ECONML = True
except ImportError:
    HAS_ECONML = False
    warnings.warn("econml not installed; using dummy model")

def train_double_ml(X, T, Y, model_y="forest", model_t="forest", cv=3):
    if not HAS_ECONML:
        return None
    if model_y == "forest":
        model_y = RandomForestRegressor(n_estimators=50, min_samples_leaf=10, random_state=42)
    else:
        model_y = RidgeCV(alphas=[0.1, 1.0, 10.0])
    if model_t == "forest":
        model_t = RandomForestRegressor(n_estimators=50, min_samples_leaf=10, random_state=42)
    else:
        model_t = RidgeCV(alphas=[0.1, 1.0, 10.0])
    dml = LinearDML(model_y=model_y, model_t=model_t, discrete_treatment=False, cv=cv, random_state=42)
    dml.fit(Y, T, X=X)
    return dml

def predict_cate(model, X):
    if model is None:
        return np.zeros(X.shape[0])
    return model.effect(X)
