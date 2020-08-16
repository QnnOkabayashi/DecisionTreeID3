from math import log
from time import time
from typing import *
from csv_reader import Case

AttrOpt = str
AttrName = str

datasets = [
    'tennis',
    'titanic',
    'breast-cancer',
    'congress84',
    'primary-tumor',
    'mushrooms'
]

idataset = 'titanic'
percent_training = 0.9  # Choose what percantage of cases to use as training
show_tree = False  # Choose whether to display the decision tree or not

class Node:
    def __init__(self, name: AttrName, parent_opt: AttrOpt):
        self.name = name  # Either a finalized category or an attribute name. Ex. 'weather'
        self.parent_opt = parent_opt  # Holds the status of the parent node. Ex. 'sunny'
        self.children: Dict[AttrOpt, Node] = {}  # Key: opt, value: child node with that attribute

    def leaf(self) -> bool:
        return not self.children


class DecisionTree:

    def __init__(self, names: List[AttrName], opts: List[Set[AttrOpt]]):
        self.category = names[0]
        self.attrs = dict(zip(names, opts))
        self.name_to_idx = dict(zip(names, range(len(names))))
        self.root = Node('root', None)


    def train(self, training_cases: Iterable[Case]) -> Node:
        usable_attrs = list(self.attrs)[1:]  # exclude category from usable attributes
        self.root = self.id3(training_cases, usable_attrs, '')
        return self.root


    def test(self, testing_cases: Iterable[Case]) -> float:
        return sum(
            self.climb(self.root, case, 0) for case in testing_cases
        ) / len(testing_cases)


    def entropy(self, remaining: Iterable[Case]) -> float:
        # count the number of yes cases and no cases from the remaining list of cases
        if not remaining:
            return 0
        categories = self.attrs[self.category]  # {'yes', 'no'}

        cat_counts = (sum(case[0] == cat for case in remaining) for cat in categories)
        
        cat_percents = map(lambda count: count / len(remaining), cat_counts)

        # log(0) is invalid, so only accumulate ones where percent > 0
        return -sum(percent * log(percent, 2) for percent in cat_percents if percent > 0)


    def gain(self, remaining: Iterable[Case], attr_name: AttrName) -> Tuple[float, Dict[AttrOpt, List[Case]]]:
        '''returns the gain when splitting by a given attribute, and the rosters it yields'''
        opts = self.attrs[attr_name]  # {'sunny', 'rainy', 'overcast'}
        opt_rosters: Dict[AttrOpt, List[Case]] = {opt: [] for opt in opts}
        for case in remaining:
            case_opt = case[self.name_to_idx[attr_name]]
            opt_rosters[case_opt].append(case)

        # print(opt_rosters.keys())

        gain: float = self.entropy(remaining) - sum(
            len(roster) / len(remaining) * self.entropy(roster) for roster in opt_rosters.values()
        )
        # self.entropy(roster) is always 0?

        return gain, opt_rosters


    def mode_category(self, remaining: List[Case]) -> AttrOpt:
        categories = [case[0] for case in remaining]
        return max(set(categories), key=categories.count)


    def id3(self, remaining: List[Case], usable_attrs: Iterable[AttrName], parent_opt: AttrOpt) -> Node:
        if all(case[0] == remaining[0][0] for case in remaining):
            category: AttrOpt = remaining[0][0]
            return Node(category, parent_opt)
        if not usable_attrs:
            # no more attributes to divide by
            # choose the attribute that is the most common amoung remaining cases
            category: AttrOpt = self.mode_category(remaining)
            return Node(category, parent_opt)

        # Calculate which attribute yields the largest gain
        top_attr: AttrName
        top_gain: float = -1
        top_rosters: Dict[AttrOpt, List[Case]]  # contains all cases with a given attr opt of the best attr
        for attr in usable_attrs:
            gain, rosters = self.gain(remaining, attr)
            if gain > top_gain :
                top_attr = attr
                top_gain = gain
                top_rosters = rosters

        node = Node(top_attr, parent_opt)
        for attr_opt, roster in top_rosters.items():  # ('sunny', [... cases ...])
            if len(roster) > 0:
                remaining_attrs = list(usable_attrs)
                remaining_attrs.remove(top_attr)
                node.children[attr_opt] = self.id3(roster, remaining_attrs, attr_opt)
            else:
                # no cases have attr_opt
                category = self.mode_category(remaining)
                node.children[attr_opt] = Node(category, attr_opt)
        return node


    def printer(self, node, depth=0) -> None:  # Recursive function that prints out a tree
        if node.leaf():
            print(f"{'|  ' * depth}> {node.name}")
        else:
            print(f"{'|  ' * depth}{node.name}?")
            for child in node.children.values():
                print(f"{'|  ' * (depth + 1)}[{child.parent_opt}]")
                self.printer(child, depth + 2)

    def __str__(self):
        def str_rec(node, depth: int) -> str:  # Recursive function that prints out a tree
            if node.leaf():
                return f"{'|  ' * depth}> {node.name}"
            else:
                lines = [f"{'|  ' * depth}{node.name}?"]
                for child in node.children.values():
                    lines.append(f"{'|  ' * (depth + 1)}[{child.parent_opt}]")
                    lines.append(f"{str_rec(child, depth + 2)}")
                return "\n".join(lines)
        
        return str_rec(node=self.root, depth=0)


    def climb(self, node, case, depth: int) -> bool:  # Recursive function to determine if a case is categorized correctly. "Climbs" the tree
        if node.leaf():
            return case[0] == node.name
        else:
            # need dictionary that maps attr_names to indicies
            next_attr_opt = case[self.name_to_idx[node.name]]
            return self.climb(node.children[next_attr_opt], case, depth + 1)


    def trim(self) -> Tuple[bool, AttrOpt]:
        '''returns if all children have same category, and that category ('None' if false)'''
        pass


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
    main(f"datasets/{idataset}.txt", percent_training, show_tree)
