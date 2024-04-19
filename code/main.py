"""
Driver file and some helper functions
Author: Kilian Jakstis
"""

import os
import argparse
from util import Observation
from decision_tree import DecisionTree
from ada_boost import AdaBoost

def handle_args():
    """
    Parse args and make appropriate routine calls
    """
    parser = argparse.ArgumentParser(description='Lab 3')
    subparsers = parser.add_subparsers(help='sub-command help')
    # learn model parser
    parser_mode1 = subparsers.add_parser('train', help='train model')
    parser_mode1.add_argument('examples', help='file with labeled examples')
    parser_mode1.add_argument('hypothesis_out', help='filepath to save hypothesis object')
    parser_mode1.add_argument('learning_type',
                              help='dt - decision tree, ada - aba boost with decision stubs')
    parser_mode1.set_defaults(func=train_routine)
    # predict model parser
    parser_mode2 = subparsers.add_parser('predict', help='predict model')
    parser_mode2.add_argument('hypothesis', help='file with hypothesis object')
    parser_mode2.add_argument('file', help='fill with observation to classify')
    parser_mode2.set_defaults(func=predict_routine)
    # parse
    args = parser.parse_args()
    args.func(args)

def train_routine(args):
    """
    Run training routine
    * tree max depth / ada boost number of stumps is passed into model.train as second arg, otherwise uses defaults
    :param args: example file path, model out path, DT/ADA mode
    """
    if os.path.isfile(args.examples):
        if args.learning_type != "dt" and args.learning_type != "ada":
            print("Learning type not recognized")
        observations = Observation.get_observations(args.examples, 1)
        model = DecisionTree() if args.learning_type == "dt" else AdaBoost()
        model.train(observations)
        model.write_to_file(args.hypothesis_out)
    else:
        print("Example data file not found.")

def predict_routine(args):
    """
    Run prediction routine
    :param args: hypothesis file, prediction examples file path
    """
    with open(args.hypothesis, 'r') as file:
        data = file.read()
    model = DecisionTree() if data[0] == "{" else AdaBoost()
    model.from_json(data)
    examples_to_classify = Observation.get_observations(args.file, 0)
    for e in examples_to_classify:
        print(model.predict(e))

if __name__ == '__main__':
    handle_args()
