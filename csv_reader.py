from typing import *
from random import shuffle
from datasets import Dataset
from decision_tree import DecisionTree, Case

class CSVReader:

    def __init__(self, dataset: Dataset):
        self.names: List[AttrName]
        self.opts: List[Set[AttrOpt]]
        self.cases: List[Case]

        filepath = f"datasets/{dataset.value}.txt"

        with open(filepath, 'r') as f:
            self.names = f.readline().strip().split(',')
            self.opts = [set() for _ in self.names]
            self.cases = [list(line.strip().split(',')) for line in f]

            for case in self.cases:
                for opt, case_opt in zip(self.opts, case):
                    opt.add(case_opt)


    def partition(self, percent_training: float) -> Tuple[List[Case], List[Case]]:
        if 0 < percent_training < 1:
            num_training = int(len(self.cases) * percent_training)
            if num_training < 1:
                raise ValueError(f"{n * 100}% training yields 0 training cases. Use a larger % training value.")
            shuffle(self.cases)
            training: List[Cases] = self.cases[:num_training]
            testing: List[Cases] = self.cases[num_training:]

            return training, testing
        else:
            raise ValueError(f"percent_error must be between 0 and 1")


    def build_tree(self) -> DecisionTree:
        return DecisionTree(self.names, self.opts)

