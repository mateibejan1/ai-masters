import csv
import os
from utils import *
import numpy as np
from random import randrange
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc, roc_auc_score
from itertools import cycle
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn import metrics


class Sklearn:
    name = None
    dataset = None
    model = None

    def __init__(self, dataset, model_name):
        self.name = model_name
        print("Creating the model (%s) ..." % self.name)
        self.dataset = dataset

    def train_score(self):
        print("Train score (%s) ..." % self.name)
        return self.model.score(self.dataset.train_input, self.dataset.train_output)

    def validation_score(self):
        print("Validation score (%s) ..." % self.name)
        return self.model.score(self.dataset.valid_input, self.dataset.valid_output)

    def test_score(self):
        print("Test score (%s) ..." % self.name)
        return self.model.score(self.dataset.test_input, self.dataset.test_output)

    def score(self, input_data, output_data):
        print("Score (%s) ..." % self.name)
        return self.model.score(input_data, output_data)

    def predict(self, input_data):
        print("Predict (%s) ..." % self.name)
        return self.model.predict(input_data)

    def decision_function(self, input_data):
        print("Roc auc curve (%s) ..." % self.name)
        return self.model.decision_function(input_data)

    def roc_auc_curve(self, input_data, output_data, strategy=""):
        y_pred_proba = self.model.predict_proba(input_data)[::, 1]
        fpr, tpr, _ = metrics.roc_curve(output_data, y_pred_proba)
        auc = metrics.roc_auc_score(output_data, y_pred_proba)
        name = self.name + strategy
        pretty_roc_auc_curve(fpr=fpr, tpr=tpr, roc_auc=auc, file_name=name)
        # plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
        # plt.legend(loc=4)
        # plt.savefig('./results/' + self.name + 'roc_curve.png')
        # plt.show()
        # # Compute ROC curve and ROC area for each class
        # output_score = self.decision_function(input_data)
        # # output_score = output_score.ravel()
        # output_data = output_data.ravel()
        # output_score = np.concatenate((1 - output_score, output_score), axis=0) ##????
        # print("output score" + str(output_score))
        # fpr = dict()
        # tpr = dict()
        # roc_auc = dict()
        # print("inainte de for")
        # for i in range(2):
        #     print(output_data[i])
        #     fpr[i], tpr[i], _ = roc_curve(output_data[i], output_score[:, i])
        #     print("fpr ", fpr[i])
        #     print("tpr ", tpr[i])
        #     roc_auc[i] = auc(fpr[i], tpr[i])
        #     print("roc_auc ", roc_auc[i])
        # print("dupa for")
        #
        # # Compute micro-average ROC curve and ROC area
        # fpr["micro"], tpr["micro"], _ = roc_curve(output_data, output_score.ravel())
        # print("line 69")
        # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # print("line 71")
        # file_name = "%s_%s_%s.png" % (self.name, strategy, str(randrange(100, 1000)))
        # print("file name " + file_name)
        # pretty_roc_auc_curve(fpr, tpr, roc_auc, file_name)

    def confusion_matrix(self, input_data, output_data, strategy="", predicted=None, action=""):
        print("Confusion matrix (%s) ..." % self.name)
        correct_values = 0
        matrix = [[0, 0], [0, 0]]
        if predicted is None:
            predicted = self.predict(input_data)
        for i in range(len(predicted)):
            if int(predicted[i]) == int(output_data[i]):
                correct_values += 1
                if int(predicted[i]) == 1:
                    matrix[0][0] += 1
                else:
                    matrix[1][1] += 1
            else:
                if int(predicted[i]) == 1:
                    matrix[0][1] += 1
                else:
                    matrix[1][0] += 1

        score = (correct_values * 100) / len(predicted)
        file_name = "%s_%s_%s_%s_%s.png" % (self.name, str(score), strategy, action, str(randrange(100, 1000)))
        with open('./results/results.txt', 'a') as f:
            f.write("%s %s\n" % (file_name, str(matrix)))
        pretty_confusion_matrix(matrix, file_name=file_name)
        # pretty_confusion_matrix(conf_matrix)

    def metrics(self, input_data, output_data, strategy, future_data, predicted=None, action=""):
        print("Metrics for (%s) ..." % self.name)
        if predicted is None:
            predicted = self.predict(input_data)

        future_predicted = self.predict(future_data)

        precision, recall, f_score, _ = precision_recall_fscore_support(output_data, predicted, pos_label=1,
                                                                        average='binary')
        metrics = {"Precision": precision,
                   "Recall": recall,
                   "F Score": f_score}
        for metric in metrics:
            print(metric, '=', metrics[metric])

        csv_rows = ['Name', "Strategy", "Action", "Accuracy", "Precision", "Recall", "F Score", "Predicted Insolvency"]
        metrics['Name'] = self.name
        metrics['Strategy'] = strategy
        metrics['Action'] = action
        metrics['Accuracy'] = (predicted == output_data).mean() * 100
        metrics['Predicted Insolvency'] = (future_predicted == 1).mean() * 100

        if not os.path.exists('results/results.csv'):
            with open('results/results.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, csv_rows)
                writer.writeheader()
                writer.writerow(metrics)
        else:
            with open('results/results.csv', 'a', newline='') as f:
                writer = csv.DictWriter(f, csv_rows)
                writer.writerow(metrics)
