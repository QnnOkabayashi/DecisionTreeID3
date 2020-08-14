from math import log
from time import time
from random import shuffle
from typing import *

Case = List[str]

# cases: Case = []  # each line in the .txt file is a case. Each field is accessed by index
# attributes: List[str] = []  # Holds attribute names so they can be accessed by index for printing. An attribute is a header
# statuses: List[Set[str]] = []  # A list of attribute statuses, with the index representing the attribute.

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

class DecisionTree:

    def __init__(self, filepath: str):
        self.categories: Set[str]  # the categories that all cases fall into
        self.cases: List[Case]
        self.attrs: Dict[str, Set[str]]
        self.attr_names: List[str]
        self.attr_enums: List[Set[str]]
        with open(filepath, 'r') as f:
            self.attr_names: List[str] = f.readline().strip().split(',')
            self.attr_enums: List[Set[str]] = [set()] * len(attr_names)

            self.categories = attr_enums[0]  # first enum is always the categorizing variable
            self.cases = [list(line.strip().split(',')) for line in f]

            for case in self.cases:
                for enum, option in zip(attr_enums, case):
                    enum.add(option)

            self.attrs = dict(zip(attr_names, attr_enums))


    def get_entropy(self, cases: Iterable[Case]) -> float:
        # count the number of yes cases and no cases from the given list of cases
        attribute_counts = (sum(case[0] == category for case in self.cases) for category in self.categories)  # loops through ['yes', 'no']

        nonzero_attribute_counts = filter(lambda count: count > 0, attribute_counts)

        nonzero_attribute_percents = map(lambda count: float(count) / len(self.cases), nonzero_attribute_counts)

        return -sum(percent * log(percent, 2) for percent in nonzero_attribute_percents)


    def get_gain(self, cases: List[List[str]], attr_name: str) -> Tuple[float, List[List[int]]]: # NOT TYPE 'int', I'M JUST CONFUSED
        '''calculate gain by splitting cases up by attribute at attribute_i'''
        # status_cases is a list where each list represents an attr enum, and contain all elements that match that enum for that category
        status_cases: List[List[Case]] = []  # Index: the status # of an attribute, Value: list of days
        '''
        states = attribute_states[attribute_i]
        for state in states:
        '''
        for enums in self.attrs[attr_name]:
            n = [case for case in self.cases if case[attribute_i] == status]
            status_cases.append(n)

        for enums in self.attrs[attr_name]:  # ['sunny', 'rainy', 'overcast']
            pass
            # 

        # status_cases = [[case for case in cases in case[attribute_i] == status] for status in statuses[attribute_i]]

        # status cases are the cases that have a status!
        # use filter() for that

        # status_cases = filter(lambda case: case[attribute_i] ) # ahh idk

        '''
        the gain is the entropy of everything minus the entropy of splitting up by the given attribute
        '''
        gain = (self.get_entropy(cases) - sum(float(len(status_case)) / len(cases) * self.get_entropy(status_case) for status_case in status_cases))

        return gain, status_cases


    def mode_category(self, cases: List[Case]) -> str:
        categories = [case[0] for case in cases]
        return max(set(categories), key=categories.count)


    def id3(self, cases: List[Case], usable_attrs: Iterable[str], parent_status):
        if len(set(case[0] for case in self.cases)) <= 1:
            # all same category
            category = self.cases[0][0]
            return Node(category, parent_status)
        if not usable_attrs:
            #  no more attributes to divide by
            category = self.mode_category(self.cases)
            return Node(category, parent_status)

        best_attr = ''
        best_gain = -1
        status_cases = []  # stc = status cases.
        for attr in usable_attrs:
            gain, attr_status_cases = self.get_gain(cases, attr)
            if gain > best_gain :
                best_attr, best_gain = attr, gain
                status_cases = attr_status_cases

        nl = Node(best_attr, parent_status)  # nl = non-leaf. Node that will be given children later
        for i, stat in enumerate(self.stats[best_attr]):  # Loop through statuses of best_attribute
            if len(status_cases[i]) != 0:  # If len(stc[i]) is 0, then there are no days with that attribute
                # the best attribute has been chosen, so now we want to divide among each of the sub groups
                x = list(usable_attrs) # copy constructor
                x.remove(best_attr)
                child = self.id3(status_cases[i], x, stat)
            else:
                child = Node(self.mode_category(cases), stat)  # the child is a leaf labeled with the most common category
            nl.children[stat] = child
        return nl


    def printer(self, node: Node, depth: int) -> None:  # Recursive function that prints out a tree
        if node.leaf():
            print(f"{('|  ' * depth)}> {node.label}")
        else:
            print(f"{('|  ' * depth)}{attributes[node.label]}?")
            for child in node.children.values():
                print(f"{('|  ' * (depth + 1))}[{child.status}]")
                self.printer(child, depth + 2)


    def climb(self, node, case, depth: int) -> bool:  # Recursive function to determine if a case is categorized correctly. "Climbs" the tree
        if node.leaf():
            return case[0] == node.label
        return self.climb(node.children[case[node.label]], case, depth + 1)  # Run again, but with the next node down the tree


def main(filepath: str, percent_training: float, show_tree: bool) -> None:
    # have it only pass the name, construct filepath in this function
    if 0 < percent_training < 1:
        start = time()
        tree = DecisionTree(filepath)
        num_training = int(len(tree.cases) * percent_training)
        if num_training == 0:
            raise Exception(f"{n * 100}% training yields 0 training cases. Use a larger % training value.")
        shuffle(tree.cases)  # Use random order

        # partition into training and testing cases
        training = tree.cases[:num_training]
        testing = tree.cases[num_training:]

        head = tree.id3(training, range(1, len(tree.attrs)), None)  # Create decision tree with ID3 algorithm
        acc = float(sum(tree.climb(head, case, 0) for case in testing))/len(testing)  # Finds accuracy of tree

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
