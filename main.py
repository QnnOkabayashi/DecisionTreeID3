from decision_tree import DecisionTree
from csv_reader import CSVReader
from datasets import Dataset

dataset = Dataset.CONGRESS84
percent_training = 0.5

data = CSVReader(dataset)
training, testing = data.partition(percent_training)

tree = data.build_tree()

tree.train(training)
accuracy = tree.test(testing)

print(tree)

print(f"Tree accuracy: {accuracy}")
