from .SvmAlgorithm import Svm
from .BoostingTreesAlgorithm import BoostingTrees
from .KnnAlgorithm import Knn
from .NNAlgorithm import NN
from .RandomForestAlgorithm import RandomForest


class SklearnAlgorithmUtils:
    def __init__(self):
        pass

    @staticmethod
    def factory(dataset, algorithm_name, strategy=None, predict_prob=False):

        if algorithm_name == 'svm':
            return Svm(dataset, algorithm_name, predict_prob)
        if algorithm_name == 'knn':
            return Knn(dataset, algorithm_name)
        if algorithm_name == 'nn':
            return NN(dataset, algorithm_name)
        if algorithm_name == 'random_forest':
            return RandomForest(dataset, algorithm_name)
        if algorithm_name == 'boosting_tree':
            return BoostingTrees(dataset, algorithm_name, load_from_disk=False, model_name=strategy)
        else:
            raise ValueError('Algorithm ' + algorithm_name + ' is not supported')
