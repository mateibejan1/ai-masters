import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import roc_curve, roc_auc_score
import os


def format_metric_value(x):
    if x == "":
        return ""
    if 0 < x < 1:
        return ["%.2f" % x]
    else:
        return ["%d" % x]


class Metrics:
    def __init__(self, validation_prediction, test_prediction, algorithm, strategy):
        self.validation_prediction = validation_prediction
        self.test_prediction = test_prediction
        self.algorithm = algorithm
        self.strategy = strategy
        if not os.path.exists("spark_results/%s" % self.strategy):
            os.makedirs("spark_results/%s" % self.strategy)
        self.metrics = {}

    def compute_metrics(self):
        def cohens_kappa(tp, tn, fp, fn):
            N = tp + tn + fp + fn
            Po = float(tp + tn) / N
            Pe = float(((tn + fp) * (tn + fn)) + ((fn + tp) * (fp + tp))) / (N * N)
            kappa = float(Po - Pe) / (1 - Pe)
            return kappa

        evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy"
        )
        val_accuracy = evaluator.evaluate(self.validation_prediction)
        print("Validation Error = %g" % (1.0 - val_accuracy))

        self.metrics["val_accuracy"] = val_accuracy

        test_accuracy = evaluator.evaluate(self.test_prediction)
        print("Test Error = %g" % (1.0 - test_accuracy))

        self.metrics["test_accuracy"] = test_accuracy

        correct = self.validation_prediction.filter(
            self.validation_prediction.label == self.validation_prediction.prediction
        )
        wrong = self.validation_prediction.filter(
            self.validation_prediction.label != self.validation_prediction.prediction
        )

        true_positives = correct.filter(
            self.validation_prediction.prediction == 1
        ).count()
        true_negatives = correct.filter(
            self.validation_prediction.prediction == 0
        ).count()
        false_positives = wrong.filter(
            self.validation_prediction.prediction == 1
        ).count()
        false_negatives = wrong.filter(
            self.validation_prediction.prediction == 0
        ).count()

        self.metrics["true_positives"] = true_positives
        self.metrics["true_negatives"] = true_negatives
        self.metrics["false_positives"] = false_positives
        self.metrics["false_negatives"] = false_negatives

        try:
            precision = float(true_positives) / (true_positives + false_positives)
        except Exception as e:
            precision = ""

        try:
            recall = float(true_positives) / (true_positives + false_negatives)
        except Exception as e:
            recall = ""

        try:
            specificity = float(true_negatives) / (true_negatives + false_positives)
        except Exception as e:
            specificity = ""

        try:
            f1measure = float(2 * recall * precision) / (recall + precision)
        except Exception as e:
            f1measure = ""

        try:
            cohens_kappa_coefficient = cohens_kappa(
                true_positives, true_negatives, false_positives, false_negatives
            )
        except Exception as e:
            cohens_kappa_coefficient = ""

        self.metrics["precision"] = precision
        self.metrics["recall"] = recall
        self.metrics["specificity"] = specificity
        self.metrics["f1measure"] = f1measure
        self.metrics["cohens_kappa_coefficient"] = cohens_kappa_coefficient

    def display_confusion_matrix(self):
        total = (
            self.metrics["true_positives"]
            + self.metrics["false_positives"]
            + self.metrics["true_negatives"]
            + self.metrics["false_negatives"]
        )
        matrix = (
            np.array(
                [
                    [self.metrics["true_positives"], self.metrics["false_positives"]],
                    [self.metrics["false_negatives"], self.metrics["true_negatives"]],
                ]
            )
            / total
        )

        matrix_df = pd.DataFrame(
            matrix,
            index=["Predicted positive", "Predicted negative"],
            columns=["Actual positive", "Actual negative"],
        )
        plt.figure(figsize=(10, 7))
        plt.title("Confusion matrix %s - %s" % (self.algorithm, self.strategy))
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(matrix_df, annot=True, annot_kws={"size": 16}, fmt="g")
        plt.savefig(
            "spark_results/%s/confusion_matrix_%s" % (self.strategy, self.algorithm)
        )
        plt.close()

    def display_roc(self):
        rows = self.validation_prediction.select(["label", "probability"]).collect()
        labels = []
        probs = []

        for row in rows:
            labels.append(row.label)
            probs.append(row.probability[1])

        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = roc_auc_score(labels, probs)

        plt.figure(figsize=(10, 7))
        lw = 2
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=lw,
            label="ROC curve (area = %0.2f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(
            "Receiver operating characteristic %s - %s"
            % (self.algorithm, self.strategy)
        )
        plt.legend(loc="lower right")
        plt.savefig("./spark_results/%s/roc_curve_%s" % (self.strategy, self.algorithm))
        plt.close()

    def display_metrics(self):
        df = pd.DataFrame(
            np.array(list(map(format_metric_value, self.metrics.values()))),
            index=self.metrics.keys(),
            columns=["Metric value"],
        )
        with open(
            "spark_results/%s/results_%s" % (self.strategy, self.algorithm), "w+"
        ) as f:
            f.write(str(df))

        print(df.to_string())
        try:
            self.display_confusion_matrix()
        except Exception as e:
            print("Could not display confusion matrix")

        try:
            self.display_roc()
        except Exception as e:
            print("Could not display ROC")
