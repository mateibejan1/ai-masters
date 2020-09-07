
from pyspark.ml.classification import (
    LinearSVC,
    LogisticRegression,
    MultilayerPerceptronClassifier,
    GBTClassifier,
    RandomForestClassifier,
    DecisionTreeClassifier,
)

from spark_algorithms.metrics import Metrics

class Algorithms():
    def __init__(self, dataset, strategy):
        self.dataset = dataset
        self.train, self.valid, self.test = self.dataset.train_assembled, self.dataset.valid_assembled, self.dataset.test_assembled
        self.strategy = strategy
        self.csv_rows = []

    def compute_metrics(self, model, algo_name):
        prediction_val = model.transform(self.valid)
        prediction_test = model.transform(self.test)
        metrics = Metrics(prediction_val, prediction_test, algo_name, self.strategy)
        metrics.compute_metrics()
        self.append_results(algo_name, metrics.metrics)
        metrics.display_metrics()

    def append_results(self, algo_name, metrics_dict):
        self.csv_rows.append(
            [
                algo_name,
                self.strategy,
                metrics_dict["true_positives"],
                metrics_dict["true_negatives"],
                metrics_dict["false_positives"],
                metrics_dict["false_negatives"],
                metrics_dict["val_accuracy"],
                metrics_dict["test_accuracy"],
                metrics_dict["precision"],
                metrics_dict["recall"],
                metrics_dict["f1measure"],
                metrics_dict["cohens_kappa_coefficient"],

            ]
        )

    def run_all(self):
        self.run_svc()
        self.run_lreg()
        self.run_nn()
        self.run_gbt()
        self.run_rf()
        self.run_dt()

    def run_svc(self):
        print("LinearSVC % s" % self.strategy)
        try:
            svc = LinearSVC()
            svc_model = svc.fit(self.train)
            self.compute_metrics(svc_model, "linear_svc")
        except Exception as e:
            print(e)
            print("ERROR: Could not train model LinearSvc")


    def run_lreg(self):
        print("LogisticRegression % s" % self.strategy)
        try:
            lr = LogisticRegression()
            lr_model = lr.fit(self.train)
            self.compute_metrics(lr_model, "logistic_regression")
        except Exception as e:
            print(e)
            print("ERROR: Could not train model  LogisticRegression %s" % self.strategy)

    def run_nn(self):
        print("NeuralNetwork % s" % self.strategy)
        try:
            layers = [len(self.dataset.columns), 30, 2]
            # layers = [len(full_train.columns) - 1, 30, 2]
            nn = MultilayerPerceptronClassifier(
                maxIter=200,
                tol=1e-08,
                seed=None,
                layers=layers,
                blockSize=128,
                stepSize=0.03,
                solver="l-bfgs",
            )
            nn_model = nn.fit(self.train)
            self.compute_metrics(nn_model, "neural_network")
        except Exception as e:
            print(e)
            print("ERROR: Could not train model  NeuralNetwork %s" % self.strategy)

    def run_gbt(self):
        print("GBTClassifier % s" % self.strategy)
        try:
            gbt = GBTClassifier(maxIter=10)
            gbt_model = gbt.fit(self.train)
            self.compute_metrics(gbt_model, "gradient_boosted_trees")
        except Exception as e:
            print(e)
            print("ERROR: Could not train model  GBTClassifier %s" % self.strategy)

    def run_rf(self):
        print("RandomForestClassifier % s" % self.strategy)
        try:
            rf = RandomForestClassifier()
            rf_model = rf.fit(self.train)
            self.compute_metrics(rf_model, "random_forest")
        except Exception as e:
            print(e)
            print("ERROR: Could not train model RandomForestClassifier %s" % self.strategy)

    def run_dt(self):
        print("DecisionTreeClassifier % s" % self.strategy)
        try:
            tree = DecisionTreeClassifier()
            tree_model = tree.fit(self.train)
            self.compute_metrics(tree_model, "decision_tree")
        except Exception as e:
            print(e)
            print("ERROR: Could not train model  DecisionTreeClassifier %s" % self.strategy)
