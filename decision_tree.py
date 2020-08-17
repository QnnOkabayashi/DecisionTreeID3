from id3_types import *
from math import log

class Node:
    def __init__(self, name: AttrName, parent_opt: AttrOpt):
        self.name = name  # Either a finalized category or an attribute name. Ex. 'weather'
        self.parent_opt = parent_opt  # Holds the status of the parent node. Ex. 'sunny'
        self.children: Dict[AttrOpt, Node] = {}  # Key: opt, value: child node with that attribute

    def leaf(self) -> bool:
        return not self.children

    def trim(self) -> Tuple[bool, AttrOpt]:
        if self.leaf():
            return True, self.name
        else:
            are_trimmable, categories = zip(*(child.trim() for child in self.children.values()))
            if all(are_trimmable) and len(set(categories)) == 1:
                self.name = categories[0]
                self.children = {}
                return True, self.name
            else:
                return False, None


class DecisionTree:

    def __init__(self, names: List[AttrName], opts: List[Set[AttrOpt]]):
        self.root: Node
        self.category = names[0]
        self.attrs = dict(zip(names, opts))


    def train(self, training_cases: Iterable[Case]) -> Node:
        usable_attrs = list(self.attrs)[1:]  # exclude category from usable attributes
        self.root = self.id3(training_cases, usable_attrs, '')
        self.root.trim()
        return self.root


    def test(self, testing_cases: Iterable[Case]) -> float:
        return sum(
            self.traverse(case) for case in testing_cases
        ) / len(testing_cases)


    def entropy(self, remaining: Iterable[Case]) -> float:
        # count the number of yes cases and no cases from the remaining list of cases
        if not remaining:
            return 0
            
        categories = self.attrs[self.category]  # {'yes', 'no'}

        cat_counts = (sum(case.category() == cat for case in remaining) for cat in categories)
        
        cat_percents = map(lambda count: count / len(remaining), cat_counts)

        # log(0) is invalid, so only accumulate ones where percent > 0
        return -sum(percent * log(percent, 2) for percent in cat_percents if percent > 0)


    def gain(self, remaining: Iterable[Case], attr_name: AttrName) -> Tuple[float, Dict[AttrOpt, List[Case]]]:
        '''returns the gain when splitting by a given attribute, and the rosters it yields'''
        opts = self.attrs[attr_name]  # {'sunny', 'rainy', 'overcast'}
        opt_rosters: Dict[AttrOpt, List[Case]] = {opt: [] for opt in opts}
        for case in remaining:
            opt_rosters[case[attr_name]].append(case)

        gain: float = self.entropy(remaining) - sum(
            len(roster) / len(remaining) * self.entropy(roster) for roster in opt_rosters.values()
        )

        return gain, opt_rosters


    def mode_category(self, remaining: List[Case]) -> AttrOpt:
        categories = [case.category() for case in remaining]
        return max(set(categories), key=categories.count)


    def id3(self, remaining: List[Case], usable_attrs: Iterable[AttrName], parent_opt: AttrOpt) -> Node:
        if not usable_attrs:
            category: AttrOpt = self.mode_category(remaining)
            return Node(category, parent_opt)
        if all(case.category() == remaining[0].category() for case in remaining):
            category: AttrOpt = remaining[0].category()
            return Node(category, parent_opt)

        # Calculate which attribute yields the largest gain
        top_attr, top_gain, top_rosters = max(
            ((attr, *self.gain(remaining, attr)) for attr in usable_attrs), 
            key=lambda attr_gain_rosters: attr_gain_rosters[1]
        )

        node = Node(top_attr, parent_opt)
        for attr_opt, roster in top_rosters.items():  # ('sunny', [... cases that are sunny ...])
            if len(roster) > 0:
                remaining_attrs = list(usable_attrs)
                remaining_attrs.remove(top_attr)
                node.children[attr_opt] = self.id3(roster, remaining_attrs, attr_opt)
            else:
                # no cases have attr_opt
                category = self.mode_category(remaining)
                node.children[attr_opt] = Node(category, attr_opt)
        return node


    def __repr__(self):

        if not hasattr(self, 'root'):
            return 'empty tree'

        def str_rec(node, depth: int) -> List[str]:  # Recursive function that prints out a tree
            if node.leaf():
                return [f"{'|  ' * (depth - 1)} > {node.name}"]
            else:
                lines = [f"{'|  ' * depth}{node.name}?"]
                for child in node.children.values():
                    lines.append(f"{'|  ' * (depth + 1)}[{child.parent_opt}]")
                    lines += str_rec(child, depth + 2)
                return lines
        
        return '\n'.join(str_rec(node=self.root, depth=0))


    def traverse(self, case) -> bool:

        if not hasattr(self, 'root'):
            return False

        def trav_rec(node, case, depth: int) -> bool:
            if node.leaf():
                return case.category() == node.name
            else:
                next_attr_opt = case[node.name]
                return trav_rec(node.children[next_attr_opt], case, depth + 1)
        
        return trav_rec(self.root, case, 0)

