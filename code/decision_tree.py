"""
Decision Tree class
Author: Kilian Jakstis
"""

import math
import json
from util import Node
from util import Model

class DecisionTree(Model):
    """
    Decision Tree
    """

    def __init__(self):
        """
        Initialize decision tree with None root
        """
        super().__init__()
        self.root = None
        self.weight = 1

    def predict(self, observation):
        """
        Predict an observation
        :param observation: observation object
        :return: binary classification
        """
        current = self.root
        if current is None:
            print("model not initialized")
            return None
        while True:
            if len(current.children) == 0:
                return current.value
            current = current.children[str(observation.attributes[int(current.value)])]

    def to_json(self):
        """
        :return: json representation of DT
        """
        try:
            return json.dumps({"weight": self.weight, "tree": self.root.to_dict()})
        except Exception as e:
            print("Error: ", e, "\n could not serialize decision tree")

    def from_json(self, json_text):
        """
        Initialize tree root and weight from json data
        :param json_text: DT in json format
        """
        try:
            info_dict = json.loads(json_text)
            self.root = Node.from_dict(json.loads(info_dict["tree"]))
            self.weight = info_dict["weight"]
        except Exception as e:
            print("Error: ", e, "\n could not load model")

    def write_to_file(self, file_path):
        """
        Write DT to file in JSON-like format
        :param file_path: file to write to
        """
        try:
            with open(file_path, "w") as file:
                file.write(self.to_json())
        except Exception as e:
            print("Error:", e), "\n model not written to output file"

    def train(self, examples, depth_limit=-1):
        """
        Learn DT and set root equal to result
        :param examples: observations list
        :param depth_limit: max depths of tree
        """
        if len(examples) == 0:
            return
        self.root = \
            self.learn_decision_tree(examples, [x for x in range(len(examples[0].attributes))], examples, depth_limit)

    @staticmethod
    def get_english_count_weight(examples):
        """
        Get number of english examples in the list
        :return: int count of eng
        """
        english_count = 0
        w = 0
        for x in examples:
            if x.classification == "en":
                english_count += 1
                w += x.weight
        return english_count, w

    @staticmethod
    def observations_same_class(observations):
        """
        See if all observations are same class
        :return: bool answer
        """
        english, _ = DecisionTree.get_english_count_weight(observations)
        return english == len(observations) or english == 0

    @staticmethod
    def majority_answer(observations):
        """
        Get majority answer of observations
        :return: majority label
        """
        _, english_weight = DecisionTree.get_english_count_weight(observations)
        if english_weight >= sum(e.weight for e in observations) / 2:
            return "en"
        return "nl"

    @staticmethod
    def binary_entropy(p):
        """
        Calculate binary entropy
        :param p: probability
        """
        if p <= 0 or p >= 1:
            return 0
        return (p * -1 * math.log(p, 2)) + ((1 - p) * -1 * math.log(1 - p, 2))

    @staticmethod
    def split_on(attribute_i, examples):
        """
        Split examples on the specified attribute
        :param attribute_i: i-th attribute
        :param examples: all observation objects
        """
        have_i = []
        not_have_i = []
        for x in examples:
            if x.attributes[attribute_i] == 1:
                have_i.append(x)
            else:
                not_have_i.append(x)
        return have_i, not_have_i

    @staticmethod
    def get_remainders(a, examples):
        """
        Calculate remainder
        :param a: attribute
        :param examples: current examples
        :return: remainder value
        """
        remainder = 0
        total_weight = sum(e.weight for e in examples)
        split1, split2 = DecisionTree.split_on(a, examples)
        if split1:
            split1_weight = sum(e.weight for e in split1)
            split1_ratio = split1_weight / total_weight
            _, english_weight1 = DecisionTree.get_english_count_weight(split1)
            remainder += split1_ratio * DecisionTree.binary_entropy(english_weight1 / split1_weight)
        if split2:
            split2_weight = sum(e.weight for e in split2)
            split2_ratio = split2_weight / total_weight
            _, english_weight2 = DecisionTree.get_english_count_weight(split2)
            remainder += split2_ratio * DecisionTree.binary_entropy(english_weight2 / split2_weight)
        return remainder

    @staticmethod
    def info_gain(attribute, examples):
        """
        Calculate information gain if a split occurs along an attribute
        :param attribute: attribute being split on
        :param examples: current observations
        :return: info gain
        """
        total_weight = sum(e.weight for e in examples)
        _, english_weight = DecisionTree.get_english_count_weight(examples)
        gain_entropy = DecisionTree.binary_entropy(english_weight / total_weight)
        remainder = DecisionTree.get_remainders(attribute, examples)
        return gain_entropy - remainder

    @staticmethod
    def most_important_attribute(attributes, examples):
        """
        Find the most important attribute
        :param attributes: list of available attributes
        :param examples: current examples
        :return: index of most important attribute
        """
        info_gain = {}
        for a in attributes:
            info_gain[a] = DecisionTree.info_gain(a, examples)
        sorted_gains = sorted(info_gain.items(), key=lambda v: v[1], reverse=True)
        return sorted_gains[0][0]

    @staticmethod
    def learn_decision_tree(observations, attribute_list, parent_examples, depth_limit, depth=0):
        """
        Learn DT
        :param observations: all training observations
        :param attribute_list: list of available attributes
        :param parent_examples: from parent - initially all examples
        :param depth_limit: max depth allowed for tree
        :param depth: current depth - starting at 0
        :return: DT root node
        """
        if depth == depth_limit:
            return Node(DecisionTree.majority_answer(observations))
        if len(observations) == 0 or len(attribute_list) == 0:
            return Node(DecisionTree.majority_answer(parent_examples))
        if DecisionTree.observations_same_class(observations):
            return Node(observations[0].classification)
        best_attribute = DecisionTree.most_important_attribute(attribute_list, observations)
        attribute_list.remove(best_attribute)
        has_attribute, not_has_attribute = DecisionTree.split_on(best_attribute, observations)
        tree_root = Node(str(best_attribute))
        tree_root.add_child("1", DecisionTree.learn_decision_tree(has_attribute, attribute_list.copy(), observations,
                                                                depth_limit, depth + 1))
        tree_root.add_child("0", DecisionTree.learn_decision_tree(not_has_attribute, attribute_list.copy(),
                                                                  observations, depth_limit, depth + 1))
        return tree_root
