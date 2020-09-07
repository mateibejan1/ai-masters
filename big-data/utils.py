import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


def simple_plot(x, y, x_label=None, y_label=None, title=None):

    plt.plot(x, y, 'o-')
    if title:
        plt.title(title)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)

    plt.show()


def show_feature_importances(features, scores):
    data = pd.DataFrame(data={'feature': features, 'score': scores})
    data = data.sort_values(by='score', ascending=False)
    pd.set_option("max_rows", None)
    print(data)


def pretty_confusion_matrix(matrix, file_name):

    df_cm = pd.DataFrame(matrix, index=[i for i in ['True', 'False']], columns=[i for i in ['True', 'False']])
    plt.figure(figsize=(10, 7))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g')
    plt.savefig('./results/' + file_name)
    # plt.show()


def pretty_roc_auc_curve(fpr, tpr, roc_auc, file_name):
    print("Creating roc auc curve")
    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('./results/' + file_name)
    # plt.show()
