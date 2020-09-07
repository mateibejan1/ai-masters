from sklearn_algorithms.Sklearn import Sklearn
from sklearn.model_selection import cross_val_score
from utils import *
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import pickle


class BoostingTrees(Sklearn):

    def __init__(self, dataset, name, load_from_disk=False, model_name=False):
        super().__init__(dataset, name)
        self.modelName = "boosting_tree.dat"

        # Best
        # estimator:
        # XGBClassifier(base_score=0.5, booster='dart', colsample_bylevel=0.99,
        #               colsample_bynode=0.27, colsample_bytree=0.71, eta=0.01,
        #               eval_metric='rmse', gamma=0.5, gpu_id=-1, importance_type='gain',
        #               interaction_constraints=None, learning_rate=0.00999999978,
        #               max_delta_step=1, max_depth=490, max_leaves=0, min_child_weight=1,
        #               missing=nan, monotone_constraints=None, n_estimators=100, n_jobs=0,
        #               num_parallel_tree=1, objective='reg:squarederror', random_state=0,
        #               reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=0.6,
        #               tree_method='auto', validate_parameters=False, verbosity=None)
        #
        # Best normalized gini score for 10 - fold search with 10 parameter combinations: -1.1821681709244811
        if not load_from_disk:
            self.model = XGBClassifier(base_score=0.5, booster='dart', colsample_bylevel=0.99,
                                       colsample_bynode=0.27, colsample_bytree=0.71, eta=0.01,
                                       eval_metric='rmse', gamma=0.5, gpu_id=-1, importance_type='gain',
                                       interaction_constraints=None, learning_rate=0.00999999978,
                                       max_delta_step=1, max_depth=490, max_leaves=0, min_child_weight=1,
                                       monotone_constraints=None, n_estimators=100, n_jobs=0,
                                       num_parallel_tree=1, objective='reg:squarederror', random_state=0,
                                       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=0.6,
                                       tree_method='auto', validate_parameters=False, verbosity=3)
            self.model.fit(self.dataset.train_input, self.dataset.train_output)
            pickle.dump(self.model, open(self.modelName, "wb"))
        else:
            self.model = pickle.load(open(self.modelName, 'rb'))

    def find_best_model(self):
        parameters_cv_xgb = {
            'booster': ['gbtree', 'gblinear', 'dart'],
            'eta': [0.3, 0.2, 0.1, 0.01, 0.001, 0.0005, 0.0003, 0.0001],
            'max_depth': [depth for depth in range(10, 1000, 10)],
            'max_leaves': [i for i in range(8)],
            'max_delta_step': [i for i in range(10)],
            'gamma': [i / 2 for i in range(1, 20)],
            'subsample': [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [sample / 100 for sample in range(10, 100, 1)],
            'colsample_bylevel': [sample / 100 for sample in range(10, 100, 1)],
            'colsample_bynode': [sample / 100 for sample in range(10, 100, 1)],
            'eval_metric': ['rmse', 'mae', 'logloss'],
            'objective': ['reg:squarederror'],
            'tree_method': ['auto', 'exact']
        }

        xgb_model = XGBClassifier()

        kf = KFold(n_splits=10, shuffle=False, random_state=1001)

        random_search = RandomizedSearchCV(xgb_model,
                                           param_distributions=parameters_cv_xgb,
                                           n_iter=10,
                                           scoring='neg_mean_squared_error',
                                           n_jobs=100,
                                           cv=kf.split(self.dataset.train_input, self.dataset.train_output),
                                           verbose=10,
                                           random_state=1001)

        random_search.fit(self.dataset.train_input, self.dataset.train_output)

        print('Best estimator:')
        print(random_search.best_estimator_)
        print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (10, 10))
        print(random_search.best_score_ * 2 - 1)
        print('\n Best hyperparameters:')
        print(random_search.best_params_)

    def test_results(self):
        scores = cross_val_score(self.model, self.dataset.input_data, self.dataset.output_data, cv=5)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        # y_pred = self.model.predict( self.dataset.input_data)
        # predictions = [round(value) for value in y_pred]
        # accuracy = accuracy_score(self.dataset.output_data, predictions)
        # print("Accuracy: %.2f%%" % (accuracy * 100.0))


        #scores = cross_val_score(self.model, self.dataset.input_data, self.dataset.output_data, cv=5)
        #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

