from sklearn.ensemble import RandomForestClassifier
from sklearn_algorithms.Sklearn import Sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV


class RandomForest(Sklearn):

    def __init__(self, dataset, name):
        super().__init__(dataset, name)
        self.model = RandomForestClassifier(
            bootstrap=True, class_weight=None,
            criterion='gini', max_depth=460, max_features=32,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=7, min_samples_split=52,
            min_weight_fraction_leaf=0.0, n_estimators=1450,
            n_jobs=None, oob_score=False, random_state=42, verbose=0,
            warm_start=False
        )
        self.model.fit(self.dataset.train_input, self.dataset.train_output)

    def get_feature_importances(self):
        return self.model.feature_importances_

    def cross_validation(self):
        model = RandomForestClassifier(
            bootstrap=True, class_weight=None,
            criterion='gini', max_depth=460, max_features=32,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=7, min_samples_split=52,
            min_weight_fraction_leaf=0.0, n_estimators=1450,
            n_jobs=None, oob_score=False, random_state=42, verbose=0,
            warm_start=False
        )
        scores = cross_val_score(model, self.dataset.input_data, self.dataset.output_data, cv=5)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    def find_best(self):
        parameters_cv_rf = {
            'bootstrap': [True],
            'n_estimators': [trees for trees in range(100, 2000, 50)],
            'max_depth': [depth for depth in range(10, 500, 10)],
            'min_samples_leaf': range(2, 100, 5),
            'min_samples_split': [sample for sample in range(2, 500, 50)],
            'max_features': [degree_freedom for degree_freedom in range(2, 37, 5)]
        }

        # instantiate model and perform random hyperparameter search with K-fold cross validation

        rf_model = RandomForestClassifier(random_state=42)

        kf = KFold(n_splits=10, shuffle=False, random_state=1001)

        random_search = RandomizedSearchCV(rf_model,
                                           param_distributions=parameters_cv_rf,
                                           n_iter=2,
                                           scoring='neg_mean_squared_error',
                                           n_jobs=20,
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