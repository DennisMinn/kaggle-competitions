import numpy as np
from sklearn.metrics import cohen_kappa_score

a = 2.998
b = 1.092

def quadratic_weighted_kappa_metric(y_true, y_pred):
    y_true = y_true + a
    y_pred = (y_pred + a).clip(1, 6).round()
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return 'QWK', qwk, True

def quadratic_weighted_kappa_objective(y_true, y_pred):
    labels = y_true + a
    preds = y_pred + a
    preds = preds.clip(1, 6)
    f = 1/2 * np.sum((preds - labels)**2)
    g = 1/2*np.sum((preds - a)**2 + b)
    df = preds - labels
    dg = preds - a
    grad = (df/g - f * dg / g**2) * len(labels)
    hess = np.ones(len(labels))
    return grad, hess

def quadratic_weighted_kappa_wrapper(task):
    assert task in ['regression', 'classification']

    def quadratic_weighted_kappa_score(eval_pred):
        predictions, labels = eval_pred
        if task == 'regression':
            quadratic_weighted_kappa = cohen_kappa_score(labels, predictions.clip(0,5).round(0), weights='quadratic')
        else:
            quadratic_weighted_kappa = cohen_kappa_score(labels, predictions.argmax(-1), weights='quadratic')
        results = {
            'quadratic_weighted_kappa': quadratic_weighted_kappa
        }
        return results

    return quadratic_weighted_kappa_score