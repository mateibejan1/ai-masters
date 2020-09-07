from data.dataset import CompanyDataset
from sklearn_algorithms.SklearnAlgorithmUtils import SklearnAlgorithmUtils

strategies = [
    'tibi_mean',
    'tibi_median',
    'knn_scaled',
    'knn_normalized',
    'median_scaled',
    'median_norm',
    'mean_scaled',
    'mean_norm',
    'soft_norm'
]

actions = [
    # 'full',
    'drop'
]

algorithms = [
    'svm',
    'knn',
    'nn',
    'boosting_tree',
    'random_forest'
]

do_we_calculate_roc = True

for strategy in strategies:
    # for action in actions:
    action = 'drop'
    dataset = None

    try:
        dataset = CompanyDataset(0.27, 0.055, strategy=strategy, action=action)
    except Exception as e:
        print("skipping strategy %s" % strategy)
        print("Error: ", e)
        continue

    for algorithm_name in algorithms:
        print("Strategy = %s Action = %s Model = %s" % (strategy, action, algorithm_name))
        try:
            model = SklearnAlgorithmUtils.factory(dataset, algorithm_name, strategy,
                                                  predict_prob=do_we_calculate_roc)
            if do_we_calculate_roc:
                model.roc_auc_curve(dataset.test_input, dataset.test_output, strategy=strategy)
            else:
                predicted = model.predict(dataset.test_input)
                model.confusion_matrix(dataset.test_input, dataset.test_output, strategy, predicted=predicted,
                                       action=action)
                model.metrics(dataset.test_input, dataset.test_output, strategy, dataset.future_data,
                              predicted=predicted, action=action)

        except Exception as e:
            print("An error occurred while testing strategy: %s and model: %s. Error: %s" %
                  (strategy, algorithm_name, str(e)))
