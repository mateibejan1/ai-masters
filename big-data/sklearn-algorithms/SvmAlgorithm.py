from sklearn.svm import SVC
from sklearn_algorithms.Sklearn import Sklearn
from utils import *
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV


class Svm(Sklearn):

    def __init__(self, dataset, name, predict_prob=False):
        super().__init__(dataset, name)
        if predict_prob:
            self.model = SVC(
                C=66, cache_size=200, class_weight=None, coef0=0.0,
                decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                max_iter=-1, probability=True, random_state=42, shrinking=True, tol=0.001,
                verbose=True
            )
        else:
            self.model = SVC(
                C=66, cache_size=200, class_weight=None, coef0=0.0,
                decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                max_iter=-1, probability=False, random_state=42, shrinking=True, tol=0.001,
                verbose=True
            )

        self.model.fit(self.dataset.train_input, self.dataset.train_output)

    def cross_validation(self):
        model = SVC(
            C=66, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
            max_iter=-1, probability=False, random_state=42, shrinking=True, tol=0.001,
            verbose=False
        )
        scores = cross_val_score(model, self.dataset.input_data, self.dataset.output_data, cv=5)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    def find_best(self):
        # define the parameter map, each value in the value list will be tested as a potential hyperparameter
        parameters_cv_rf = {
            'C': [C for C in range(1, 101)],
            'gamma': [gamma for gamma in ['scale', 'auto']],
            'kernel': [kernel for kernel in ['linear', 'poly', 'rbf', 'sigmoid']]
        }

        # instantiate model and perform random hyperparameter search with K-fold cross validation

        svm_model = SVC(random_state=42)

        kf = KFold(n_splits=10, shuffle=False, random_state=1001)

        random_search = RandomizedSearchCV(svm_model,
                                           param_distributions=parameters_cv_rf,
                                           n_iter=3,
                                           scoring='neg_mean_squared_error',
                                           n_jobs=10,
                                           cv=kf.split(self.dataset.input_data, self.dataset.output_data),
                                           verbose=10,
                                           random_state=1001)

        random_search.fit(self.dataset.input_data, self.dataset.output_data)

        # print best features

        print('Best estimator:')
        print(random_search.best_estimator_)
        print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (10, 10))
        print(random_search.best_score_ * 2 - 1)
        print('\n Best hyperparameters:')
        print(random_search.best_params_)
