from id3_types import *
from glob import glob
from random import shuffle
from decision_tree import DecisionTree

class Dataset:

    def __init__(self, dataset_name: str):
        self.names: List[AttrName]
        self.opts: List[Set[AttrOpt]]
        self.cases: List[Case]

        filepath = f"datasets/{dataset_name}.txt"

        with open(filepath, 'r') as f:
            self.names = f.readline().strip().split(',')
            class Case:
                name_to_idx = dict(zip(self.names, range(len(self.names))))

                def __init__(self, opts: Iterable[AttrOpt]):
                    self.opts = list(opts)

                def __getitem__(self, name: AttrName) -> AttrOpt:
                    return self.opts[Case.name_to_idx[name]]

                def __iter__(self) -> Iterator[AttrOpt]:
                    return self.opts.__iter__()

                def category(self) -> AttrOpt:
                    return self.opts[0]

            self.opts = [set() for _ in self.names]
            self.cases = [Case(line.strip().split(',')) for line in f]

            for case in self.cases:
                for opt, case_opt in zip(self.opts, case):
                    opt.add(case_opt)


    @classmethod
    def from_user(cls):
        datasets = list(map(
            lambda path: path.split('/')[1].rstrip('.txt'),
            glob("datasets/*.txt")
        ))
        choice = -1
        while choice < 0 or choice > len(datasets):
            for i, dataset in enumerate(datasets):
                print(f"{i}) {dataset}")
            try:
                choice = int(input("Select a dataset: "))
                if 0 <= choice < len(datasets):
                    return Dataset(datasets[choice])
                else:
                    print(f"Selection must be in range 0 - {len(datasets) - 1}")
            except ValueError:
                print("Selection must be an integer")


    def partition(self, percent_training: float) -> Tuple[List[Case], List[Case]]:
        if 0 < percent_training <= 1:
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

