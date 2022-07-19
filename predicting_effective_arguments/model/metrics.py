from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score)


def accuracy(y_true, y_pred):
    return accuracy_score(y_true.cpu(), y_pred.cpu())


def precision(y_true, y_pred):
    return precision_score(y_true.cpu(), y_pred.cpu(), average="micro")


def recall(y_true, y_pred):
    return recall_score(y_true.cpu(), y_pred.cpu(), average="micro")


def f1(y_true, y_pred):
    return f1_score(y_true.cpu(), y_pred.cpu(), average="micro")
