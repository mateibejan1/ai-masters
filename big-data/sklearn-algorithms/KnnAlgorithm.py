from sklearn.neighbors import KNeighborsClassifier
from sklearn_algorithms.Sklearn import Sklearn
from sklearn.model_selection import cross_val_score
from utils import *
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV


class Knn(Sklearn):

    def __init__(self, dataset, name):
        super().__init__(dataset, name)
        self.model = KNeighborsClassifier(
            algorithm='auto', leaf_size=30, metric='minkowski',
            metric_params=None, n_jobs=None, n_neighbors=96, p=2,
            weights='distance'
        )
        self.model.fit(self.dataset.train_input, self.dataset.train_output)

    def cross_validation(self):
        model = KNeighborsClassifier(
            algorithm='auto', leaf_size=30, metric='minkowski',
            metric_params=None, n_jobs=None, n_neighbors=96, p=2,
            weights='distance'
        )
        scores = cross_val_score(model, self.dataset.input_data, self.dataset.output_data, cv=5)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    def find_best(self):
        parameters_cv_rf = {
            'n_neighbors': [K for K in range(2, 100)],
            'weights': [weights for weights in ['uniform', 'distance']],
            'algorithm': [algorithm for algorithm in ['auto', 'ball_tree', 'kd_tree', 'brute']],
        }

        # instantiate model and perform random hyperparameter search with K-fold cross validation

        rf_model = KNeighborsClassifier()

        kf = KFold(n_splits=10, shuffle=False, random_state=1001)

        random_search = RandomizedSearchCV(rf_model,
                                           param_distributions=parameters_cv_rf,
                                           n_iter=2,
                                           scoring='neg_mean_squared_error',
                                           n_jobs=5,
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

