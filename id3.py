from math import log
from time import time
from random import shuffle
from typing import *

Case = List[str]

cases: Case = []  # each line in the .txt file is a case. Each field is accessed by index
attributes: List[str] = []  # Holds attribute names so they can be accessed by index for printing. An attribute is a header
statuses: List[Set[str]] = []  # A list of attribute statuses, with the index representing the attribute.

datasets = [
    'tennis',
    'titanic',
    'breast-cancer',
    'congress84',
    'primary-tumor',
    'mushrooms'
]

dataset = 'titanic'
percent_training = 0.9  # Choose what percantage of cases to use as training
show_tree = False  # Choose whether to display the decision tree or not

class Node:  # Each node is an object on the tree
    def __init__(self, label, status):
        self.label = label  # Either a finalized category or an attribute name
        self.status = status  # Holds the status of the parent node. Ex. 'sunny'
        self.children = {}  # Key: status, value: child node with that attribute

    def leaf(self):
        return not self.children


def read(filepath: str) -> None:
    with open(filepath, 'r') as f:
        dataset_attrs = f.readline().strip().split(',')
        for attr in dataset_attrs:
            attributes.append(attr)
            statuses.append([])

        recent = [''] * len(dataset_attrs)

        for line_i, line in enumerate(f):
            cases.append([])
            for word_i, word in enumerate(line.strip().split(',')):
                # use a set here
                if word != recent[word_i]:
                    if word not in statuses[word_i]:  # Don't check to see if status is known, see if it's the same as the last
                        statuses[word_i].append(word)
                        recent[word_i] = word
                cases[line_i].append(word)


def entropy(cases: Iterable[Case]) -> float:
    attribute_counts = (sum(case[0] == category for case in cases) for category in statuses[0])

    nonzero_attribute_counts = filter(lambda count: count > 0, attribute_counts)

    nonzero_attribute_percents = map(lambda count: float(count) / len(cases), nonzero_attribute_counts)

    return -sum(percent * log(percent, 2) for percent in nonzero_attribute_percents)


def get_gain(cases: List[List[str]], attribute_i: int) -> Tuple[float, List[List[int]]]: # NOT TYPE 'int', I'M JUST CONFUSED
    '''calculate gain by splitting cases up by attribute at attribute_i'''
    # returns (gain, ?)
    status_cases = []  # Index: the status # of an attribute, Value: list of days
    '''
    states = attribute_states[attribute_i]
    for state in states:
    '''
    for status in statuses[attribute_i]:
        n = [case for case in cases if case[attribute_i] == status]
        status_cases.append(n if n else [])

    # status cases are the cases that have a status!
    # use filter() for that

    # status_cases = filter(lambda case: case[attribute_i] ) # ahh idk

    gain = (entropy(cases) - sum(float(len(status_case)) / len(cases) * entropy(status_case) for status_case in status_cases))

    return gain, status_cases


def mode_category(cases: List[Case]) -> str:
    categories = [case[0] for case in cases]
    return max(set(categories), key=categories.count)


def id3(cases: List[Case], usable_attrs: Iterable[str], parent_status):
    if len(set(case[0] for case in cases)) <= 1:
        # all same category
        category = cases[0][0]
        return Node(category, parent_status)
    if not usable_attrs:
        #  no more attributes to divide by
        category = mode_category(cases)
        return Node(category, parent_status)

    best_attr = ''
    best_gain = -1
    status_cases = []  # stc = status cases.
    for attr in usable_attrs:
        gain, attr_status_cases = get_gain(cases, attr)
        if gain > best_gain :
            best_attr, best_gain = attr, gain
            status_cases = attr_status_cases

    nl = Node(best_attr, parent_status)  # nl = non-leaf. Node that will be given children later
    for i, stat in enumerate(statuses[best_attr]):  # Loop through statuses of best_attribute
        if len(status_cases[i]) != 0:  # If len(stc[i]) is 0, then there are no days with that attribute
            # the best attribute has been chosen, so now we want to divide among each of the sub groups
            x = list(usable_attrs) # copy constructor
            x.remove(best_attr)
            child = id3(status_cases[i], x, stat)
        else:
            child = Node(mode_category(cases), stat)  # the child is a leaf labeled with the most common category
        nl.children[stat] = child
    return nl


def printer(node: Node, depth: int) -> None:  # Recursive function that prints out a tree
    if node.leaf():
        print(f"{('|  ' * depth)}> {node.label}")
    else:
        print(f"{('|  ' * depth)}{attributes[node.label]}?")
        for child in node.children.values():
            print(f"{('|  ' * (depth + 1))}[{child.status}]")
            printer(child, depth + 2)


def climb(node, case, depth: int) -> bool:  # Recursive function to determine if a case is categorized correctly. "Climbs" the tree
    if node.leaf():
        return case[0] == node.label
    return climb(node.children[case[node.label]], case, depth + 1)  # Run again, but with the next node down the tree


def main(filepath: str, percent_training: float, show_tree: bool) -> None:
    if 0 < percent_training < 1:
        start = time()
        read(filepath)
        num_training = int(len(cases) * percent_training)
        if num_training == 0:
            raise Exception(f"{n * 100}% training yields 0 training cases. Use a larger % training value.")
        shuffle(cases)  # Use random order

        # partition into training and testing cases
        training = cases[:num_training]
        testing = cases[num_training:]

        tree = id3(training, range(1, len(attributes)), None)  # Create decision tree with ID3 algorithm
        acc = float(sum(climb(tree, c, 0) for c in testing))/len(testing)  # Finds accuracy of tree

        period = time() - start

        print('\nDataset: ' + filepath.split('/')[1].rstrip('.txt') + '\n'*2 + 'Training with ' + str(len(training)) + ' cases\nTesting with ' + str(len(testing)) + ' cases\nTime: ' + str(period) + 's\nAccuracy: ' + str(acc) + '\n')
        if show_tree:
            printer(tree, 0)

    elif percent_training == 1:
        start_time = time()
        read(filepath)  # Have to run read() first to fill cases[]
        tree = id3(cases, range(1, len(attributes)), None)  # Create decision tree with ID3 algorithm
        end_time = time()
        print('\nDataset: ' + filepath.split('/')[1].rstrip('.txt') + '\n' * 2 + 'Time: ' + str(end_time - start_time) + 's\n')
        if show_tree:
            printer(tree, 0)

    else:
        raise ValueError(f"Percent training ({percent_training}) must be in range 0 < n <= 1")

if __name__== "__main__":
    main(f"datasets/{dataset}.txt", percent_training, show_tree)
