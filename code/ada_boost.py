"""
Decision Stump AdaBoost Class
Author: Kilian Jakstis
"""

import json
from decision_tree import DecisionTree
from util import Observation
import math

class AdaBoost:
    """
    AdaBoost Model
    """

    def __init__(self):
        """
        Initial model
        """
        super().__init__()
        self.stumps = None

    def predict(self, observation):
        """
        Predict observation label using weighted input from all stumps
        :param observation: tuple to classify
        :return: estimated label
        """
        if self is None:
            print("model not initialized")
            return None
        dutch_votes = 0
        english_votes = 0
        for s in self.stumps:
            c = s.predict(observation)
            if c == "en":
                english_votes += s.weight
            else:
                dutch_votes += s.weight
        return "en" if english_votes >= dutch_votes else "nl"

    def train(self, observations, h_count=25):
        """
        Set list of decision stumps to result of ada boost alg
        :param observations: all observations
        :param h_count: number of stumps to have - default is 25
        """
        if len(observations) == 0:
            return
        self.stumps = AdaBoost.learn_stumps(observations, h_count)

    def from_json(self, json_text):
        """
        Initialize stumps from json formatted data
        :param json_text: json data
        """
        trees = []
        try:
            stump_data = json.loads(json_text)
            for stump in stump_data:
                tree = DecisionTree()
                tree.from_json(stump)
                trees.append(tree)
            self.stumps = trees
        except Exception as e:
            print("Error: ", e, "\n could not deserialize adaboost model")

    def to_json(self):
        """
        :return: json representation of list of decision stumps
        """
        json_list = []
        for s in self.stumps:
            json_list.append(s.to_json())
        try:
            return json.dumps(json_list)
        except Exception as e:
            print("Error: ", e, "\n could not serialize adaboost model")
            return None

    def write_to_file(self, filepath):
        """
        Write json formatted ada boost model to file
        """
        try:
            with open(filepath, "w") as file:
                file.write(self.to_json())
        except Exception as e:
            print("Error: ", e, "\n could not write adaboost model to file")

    @staticmethod
    def learn_stumps(observations, hypothesis_count):
        """
        Learn stumps
        :param observations: all training examples
        :param hypothesis_count: number of hypotheses desired
        :return: list of weighted decision stumps learned
        """
        for o in observations:
            o.weight = 1 / len(observations)
        hypotheses = []
        for c in range(hypothesis_count):
            error = 0
            stump = DecisionTree()
            stump.train(observations, 1)
            for o in observations:
                if o.classification != stump.predict(o):
                    error += o.weight
            delta_weight = error / (1 - error)
            for o in observations:
                if o.classification == stump.predict(o):
                    o.weight = o.weight * delta_weight
            Observation.normalize_weights(observations)
            stump.weight = math.log(((1 - error) / error), 2) / 2 if error != 0 else 10000
            hypotheses.append(stump)
        return hypotheses
