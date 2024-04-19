"""
File of helper classes - observations, model superclass, node
Author: Kilian Jakstis
"""

import json
import re
from abc import ABC, abstractmethod

class Model(ABC):
    """
    Class for representing a model (DT or ADA)
    Ensures all models have train, write_to_file, to and from json methods
    """

    @abstractmethod
    def train(self, observations):
        pass

    @abstractmethod
    def write_to_file(self, filepath):
        pass

    @abstractmethod
    def from_json(self, json_text):
        pass

    @abstractmethod
    def to_json(self):
        pass

class Observation:
    """
    Class representing an observation
    """

    def __init__(self, attributes, classification):
        """
        Initialize the observation
        :param attributes: binary tuple representing attributes
        :param classification: english or dutch label
        """
        self.attributes = attributes
        self.classification = classification
        self.weight = 1

    @staticmethod
    def normalize_weights(observations):
        """
        Normalize the weights of all training observations
        :param observations: all observations
        """
        magnitude = sum(o.weight for o in observations)
        for o in observations:
            o.weight /= magnitude

    @staticmethod
    def extract_features(s):
        """
        Derive binary attribute tuple from string
        * 1 represents having the attribute (or Dutch articles), 1 indicates Dutch, 0 indicates English ideally
        :param s: string of text data
        :return: the binary tuple of features
        """
        dutch_articles = ["de", "het", "een"]
        english_articles = ["a", "an", "the"]
        first_letter, double_vowel, suffix, j_consonant, articles = 0, 0, 0, 0, 0
        words = s.split(" ")
        eng_article_count = 0
        dutch_article_count = 0
        suffix_count = 0
        first_letter_count = 0
        for w in words:
            if len(w) > 0:
                if w[0] in ["k", "j", "z", "v", "g"] and first_letter == 0:
                    first_letter_count += 1
                if len(w) >= 2 and suffix == 0:
                    if (w[len(w) - 2:]) == "en" or (w[len(w) - 2:]) == "ij" or (w[len(w) - 2:]) == "ig":
                        suffix_count += 1
                if "aa" in w or "uu" in w and double_vowel == 0:
                    double_vowel = 1
                if w.find("j") != -1 and j_consonant == 0:
                    if w.find("j") == len(w) - 1:
                        j_consonant = 1
                    else:
                        if w[w.find("j") + 1] not in ["a", "e", "i", "o", "u"]:
                            j_consonant = 1
                if w in dutch_articles:
                    dutch_article_count += 1
                if w in english_articles:
                    eng_article_count += 1
        articles = 1 if dutch_articles >= english_articles else 0
        suffix = 1 if suffix_count >= 2 else 0
        first_letter = 1 if first_letter_count >= 3 else 0
        return first_letter, double_vowel, suffix, j_consonant, articles

    @staticmethod
    def get_observations(path, training):
        """
        Initialize observations from data file
        :param path: observations file path
        :param training: 1 if in training mode, 0 if prediction mode
        :return: list of observation objects
        """
        observations = []
        try:
            if training:
                with open(path) as file:
                    line = file.readline()
                    while line:
                        data = line.strip().split("|")
                        if len(data) != 2:
                            print("Issue with training file format")
                            return None
                        if data[0] != "nl" and data[0] != "en":
                            print("wrong label")
                            return None
                        normalizer_string = re.sub(r'[^a-zA-Z\s]', ' ', data[1].lower())
                        features = Observation.extract_features(normalizer_string)
                        observations.append(Observation(features, "en" if data[0] == "en" else "nl"))
                        line = file.readline()
                return observations
            else:
                with open(path) as file:
                    line = file.readline().strip()
                    while line:
                        data = line.split("|")[-1]
                        s = re.sub(r'[^a-zA-Z\s]', ' ', data.lower())
                        features = Observation.extract_features(s)
                        observations.append(Observation(features, None))
                        line = file.readline().strip()
                return observations
        except Exception as e:
            print("Error: ", e, "\n could not load examples")
            return []

class Node(dict):
    """
    Class representing a node of decision tree
    """

    def __init__(self, value):
        """
        Initialize the node with a given value and empty list of children
        :param value:
        """
        super().__init__()
        self.__dict__ = self
        self.value = value
        self.children = {}

    def add_child(self, has_attribute, node):
        """
        Add a child node
        :param has_attribute: 0 or 1 edge label indicating what value of parent's attribute takes it there
        :param node: child node object
        """
        self.children[has_attribute] = node

    def to_dict(self):
        """
        :return: dump to json string
        """
        return json.dumps(self)

    @staticmethod
    def from_dict(root_dictionary):
        """
        Convert a dictionary string to a dictionary, then from dictionary to nodes recursively
        :param root_dictionary: dictionary string from file
        :return: root node of tree
        """
        n = Node(root_dictionary["value"])
        for x in root_dictionary["children"]:
            n.add_child(x, Node.from_dict(root_dictionary["children"][x]))
        return n
