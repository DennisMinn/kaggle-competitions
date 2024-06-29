import numpy as np
from sklearn.tree import DecisionTreeRegressor
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

class EssayScoringModel(AutoModelForSequenceClassification):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, task, *model_args, **kwargs):
        assert task in ['regression', 'classification']

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        if task == 'regression':
            config.attention_probs_dropout_prob = 0.0
            config.hidden_dropout_prob = 0.0
            config.num_labels = 1
        else:
            config.num_labels = 6

        model = super().from_pretrained(pretrained_model_name_or_path, config=config)
        model.task = task

        return model

    @staticmethod
    def format_predictions(task, predictions):
        if task == 'regression':
            predictions = predictions + 1
        else:
            predictions = predictions.argmax(axis=1) + 1

        predictions = np.clip(predictions, 1, 6)
        return predictions

    @staticmethod
    def create_thresholds(labels, predictions):
        thresholds = []
        for start, end in [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]:
            indices = [index for index, pred in enumerate(predictions) if start <= pred <= end]
            prediction_subset = np.array(predictions[indices])
            label_subset = np.array(labels[indices])

            prediction_subset = prediction_subset.reshape(-1, 1)

            tree = DecisionTreeRegressor(max_depth=1)
            tree.fit(prediction_subset, label_subset)
            thresholds.append(tree.tree_.threshold[0])

        return thresholds