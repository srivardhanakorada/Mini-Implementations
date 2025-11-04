import numpy as np
from collections import Counter
from sklearn import datasets, model_selection, metrics
import matplotlib.pyplot as plt

class TreeNode():
    def __init__(self, data, feature_idx, feature_val, prediction_probs, information_gain):
        self.data = data
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.prediction_probs = prediction_probs
        self.information_gain = information_gain
        self.feature_importance = self.data.shape[0] * self.information_gain  ### TODO: Revisit
        self.left = None
        self.right = None
    def node_def(self) -> str:
        if (self.left or self.right):
            return f"NODE | Information Gain = {self.information_gain} | Split IF X[{self.feature_idx}] < {self.feature_val} THEN left O/W right"
        else:
            unique_values, value_counts = np.unique(self.data[:,-1], return_counts=True)
            output = ", ".join([f"{value}->{count}" for value, count in zip(unique_values, value_counts)])            
            return f"LEAF | Label Counts = {output} | Pred Probs = {self.prediction_probs}"
        
class DecisionTree():
    def __init__(self, max_depth=4, min_samples_leaf=1, 
                 min_information_gain=0.0, numb_of_features_splitting=None,
                 amount_of_say=None) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.numb_of_features_splitting = numb_of_features_splitting
        self.amount_of_say = amount_of_say
    def _entropy(self, class_probabilities: list) -> float:
        return sum([-p * np.log2(p) for p in class_probabilities if p>0])
    def _class_probabilities(self, labels: list) -> list:
        total_count = len(labels)
        return [label_count / total_count for label_count in Counter(labels).values()]
    def _data_entropy(self, labels: list) -> float:
        return self._entropy(self._class_probabilities(labels))
    def _partition_entropy(self, subsets: list) -> float:
        total_count = sum([len(subset) for subset in subsets])
        return sum([self._data_entropy(subset) * (len(subset) / total_count) for subset in subsets])
    def _split(self, data: np.array, feature_idx: int, feature_val: float) -> tuple:
        mask_below_threshold = data[:, feature_idx] < feature_val
        group1 = data[mask_below_threshold]
        group2 = data[~mask_below_threshold]
        return group1, group2
    def _select_features_to_use(self, data: np.array) -> list:
        feature_idx = list(range(data.shape[1]-1))
        if self.numb_of_features_splitting == "sqrt":
            feature_idx_to_use = np.random.choice(feature_idx, size=int(np.sqrt(len(feature_idx))))
        elif self.numb_of_features_splitting == "log":
            feature_idx_to_use = np.random.choice(feature_idx, size=int(np.log2(len(feature_idx))))
        else:
            feature_idx_to_use = feature_idx
        return feature_idx_to_use
    def _find_best_split(self, data: np.array) -> tuple:
        min_part_entropy = 1e9
        feature_idx_to_use =  self._select_features_to_use(data)
        for idx in feature_idx_to_use:
            feature_vals = np.percentile(data[:, idx], q=np.arange(25, 100, 25))
            for feature_val in feature_vals:
                g1, g2, = self._split(data, idx, feature_val)
                part_entropy = self._partition_entropy([g1[:, -1], g2[:, -1]])
                if part_entropy < min_part_entropy:
                    min_part_entropy = part_entropy
                    min_entropy_feature_idx = idx
                    min_entropy_feature_val = feature_val
                    g1_min, g2_min = g1, g2
        return g1_min, g2_min, min_entropy_feature_idx, min_entropy_feature_val, min_part_entropy

    def _find_label_probs(self, data: np.array) -> np.array:
        labels_as_integers = data[:,-1].astype(int)
        total_labels = len(labels_as_integers)
        label_probabilities = np.zeros(len(self.labels_in_train), dtype=float)
        for i, label in enumerate(self.labels_in_train):
            label_index = np.where(labels_as_integers == i)[0]
            if len(label_index) > 0:
                label_probabilities[i] = len(label_index) / total_labels
        return label_probabilities

    def _create_tree(self, data: np.array, current_depth: int) -> TreeNode:
        if current_depth > self.max_depth:
            return None
        split_1_data, split_2_data, split_feature_idx, split_feature_val, split_entropy = self._find_best_split(data)
        label_probabilities = self._find_label_probs(data)
        node_entropy = self._entropy(label_probabilities)
        information_gain = node_entropy - split_entropy
        node = TreeNode(data, split_feature_idx, split_feature_val, label_probabilities, information_gain)
        if self.min_samples_leaf > split_1_data.shape[0] or self.min_samples_leaf > split_2_data.shape[0]:
            return node
        elif information_gain < self.min_information_gain:
            return node
        current_depth += 1
        node.left = self._create_tree(split_1_data, current_depth)
        node.right = self._create_tree(split_2_data, current_depth)
        return node
    
    def _predict_one_sample(self, X: np.array) -> np.array:
        node = self.tree
        while node:
            pred_probs = node.prediction_probs
            if X[node.feature_idx] < node.feature_val:
                node = node.left
            else:
                node = node.right
        return pred_probs

    def train(self, X_train: np.array, Y_train: np.array) -> None:
        self.labels_in_train = np.unique(Y_train)
        train_data = np.concatenate((X_train, np.reshape(Y_train, (-1, 1))), axis=1)
        self.tree = self._create_tree(data=train_data, current_depth=0)
        self.feature_importances = dict.fromkeys(range(X_train.shape[1]), 0)
        self._calculate_feature_importance(self.tree)
        self.feature_importances = {k: v / total for total in (sum(self.feature_importances.values()),) for k, v in self.feature_importances.items()}

    def predict_proba(self, X_set: np.array) -> np.array:
        pred_probs = np.apply_along_axis(self._predict_one_sample, 1, X_set)
        return pred_probs

    def predict(self, X_set: np.array) -> np.array:
        pred_probs = self.predict_proba(X_set)
        preds = np.argmax(pred_probs, axis=1)
        return preds    
        
    def _print_recursive(self, node: TreeNode, level=0) -> None:
        if node != None:
            self._print_recursive(node.left, level + 1)
            print('    ' * 4 * level + '-> ' + node.node_def())
            self._print_recursive(node.right, level + 1)

    def print_tree(self) -> None:
        self._print_recursive(node=self.tree)

    def _calculate_feature_importance(self, node):
        if node != None:
            self.feature_importances[node.feature_idx] += node.feature_importance
            self._calculate_feature_importance(node.left)
            self._calculate_feature_importance(node.right)  


if __name__ == "__main__":
    iris = datasets.load_iris()
    X = np.array(iris.data)
    Y = np.array(iris.target)
    iris_feature_names = iris.feature_names
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.25, random_state=0)
    my_tree = DecisionTree(max_depth=4, min_samples_leaf=1, min_information_gain=0)
    my_tree.train(X_train, Y_train)
    my_tree.print_tree()
    train_preds = my_tree.predict(X_set=X_train)
    print("TRAIN PERFORMANCE")
    print("Train size", len(Y_train))
    print("True preds", sum(train_preds == Y_train))
    print("Train Accuracy", sum(train_preds == Y_train) / len(Y_train))
    test_preds = my_tree.predict(X_set=X_test)
    print("TEST PERFORMANCE")
    print("Test size", len(Y_test))
    print("True preds", sum(test_preds == Y_test))
    print("Accuracy", sum(test_preds == Y_test) / len(Y_test))
    # Feature importance
    plt.bar(range(len(my_tree.feature_importances)), 
            list(my_tree.feature_importances.values()), tick_label=iris_feature_names)
    plt.title("Feature Importance")
    plt.xlabel("Feature")
    # plt.xticks(rotation=90)
    plt.ylabel("Feature Importance")
    plt.savefig("results/plots/iris_dt.png")
