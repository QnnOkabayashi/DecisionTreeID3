from decision_tree import DecisionTree
from csv_reader import CSVReader
from datasets import Dataset

dataset = Dataset.TENNIS
percent_training = 1

data = CSVReader(dataset)
training, testing = data.partition(percent_training)

tree = DecisionTree(data.names, data.opts)

# gain, rosters = tree.gain(training, data.names[1])
# print(gain, len(rosters), data.names[1])

tree.train(training)
# accuracy = tree.test(testing)

print(tree)
# tree.printer(node=tree.root)

# print(f"Tree accuracy: {accuracy}")
