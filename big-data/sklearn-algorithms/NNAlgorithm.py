from sklearn.neural_network import MLPClassifier
from sklearn_algorithms.Sklearn import Sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV


class NN(Sklearn):

    def __init__(self, dataset, name):
        super().__init__(dataset, name)
        self.model = MLPClassifier(
            activation='tanh', alpha=0.0001, batch_size='auto', beta_1=0.9,
            beta_2=0.999, early_stopping=False, epsilon=1e-08,
            hidden_layer_sizes=(30,), learning_rate='constant',
            learning_rate_init=0.001, max_iter=200,
            momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
            power_t=0.5, random_state=None, shuffle=True, solver='lbfgs',
            tol=0.0001, validation_fraction=0.1, verbose=False,
            warm_start=False)

        self.model.fit(self.dataset.train_input, self.dataset.train_output)

    def cross_validation(self):

        model = MLPClassifier(
            activation='tanh', alpha=0.0001, batch_size='auto', beta_1=0.9,
            beta_2=0.999, early_stopping=False, epsilon=1e-08,
            hidden_layer_sizes=(30,), learning_rate='constant',
            learning_rate_init=0.001, max_fun=15000, max_iter=200,
            momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
            power_t=0.5, random_state=None, shuffle=True, solver='lbfgs',
            tol=0.0001, validation_fraction=0.1, verbose=False,
            warm_start=False)
        scores = cross_val_score(model, self.dataset.input_data, self.dataset.output_data, cv=5)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    def find_best(self):
        parameters_cv_rf = {
            'activation': [activation for activation in ['identity', 'logistic', 'tanh', 'relu']],
            'solver': [solver for solver in ['lbfgs', 'sgd', 'adam']],
            'hidden_layer_sizes': [(size,) for size in range(5, 150, 5)]
        }

        # instantiate model and perform random hyperparameter search with K-fold cross validation

        rf_model = MLPClassifier()

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