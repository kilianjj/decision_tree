"""
Model evaluation helper file
Author: Kilian Jakstis
"""

from util import Observation
from decision_tree import DecisionTree
from ada_boost import AdaBoost

def test(model_path, observations_path):
    """
    Test the model and display accuracy, precision and recall among labels
    :param model_path: model file
    :param observations_path: test observation file
    """
    with open(model_path, 'r') as model_file:
        model_data = model_file.read()
    model = DecisionTree() if model_data[0] == "{" else AdaBoost()
    model.from_json(model_data)
    examples_to_classify = Observation.get_observations(observations_path, 1)
    number_english = len([x for x in examples_to_classify if x.classification == "en"])
    number_dutch = len(examples_to_classify) - number_english
    correct_english = 0
    correct_dutch = 0
    incorrect_english = 0
    incorrect_dutch = 0
    for e in examples_to_classify:
        estimated_label = model.predict(e)
        if e.classification == "en":
            if estimated_label == "en":
                correct_english += 1
            else:
                incorrect_dutch += 1
        else:
            if estimated_label == "nl":
                correct_dutch += 1
            else:
                incorrect_english += 1
    print(f"{type(model)} model:\n"
          f"Accuracy: {(correct_dutch + correct_english) / (number_dutch + number_english)}\n"
          f"English precision: {correct_english / (correct_english + incorrect_english)}\n"
          f"English recall: {correct_english / (correct_english + incorrect_dutch)}\n"
          f"Dutch precision: {correct_dutch / (correct_dutch + incorrect_dutch)}\n"
          f"Dutch recall {correct_dutch / (correct_dutch + incorrect_english)}\n")

if __name__ == "__main__":
    # test(r"C:\Users\jakst\PycharmProjects\lab3\best.model", r"C:\Users\jakst\PycharmProjects\lab3\test_data.txt")
    test(r"C:\Users\jakst\Downloads\hypothesis.txt", r"C:\Users\jakst\PycharmProjects\lab3\test_data.txt")
