import numpy as np
import tensorflow as tf

def adjust_weights(model):
    # Adjust weights for dropout
    weights = model.get_weights()
    weights[0] *= 1 / 0.4
    model.set_weights(weights)

def predict(model, X_test_norm):
    return model.predict(X_test_norm)

def mc_predict(model, X_test_norm, n_samples=10):
    # Monte Carlo prediction
    y_pred_stack = np.stack([model.predict(X_test_norm) for _ in range(n_samples)])
    return y_pred_stack
