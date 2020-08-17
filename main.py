from decision_tree import DecisionTree
from csv_reader import CSVReader
from datasets import Dataset

dataset = Dataset.PRIMARY_TUMOR
percent_training = 0.9

data = CSVReader(dataset)
training, testing = data.partition(percent_training)

tree = DecisionTree(data.names, data.opts)

tree.train(training)
accuracy = tree.test(testing)

print(tree)

print(f"Tree accuracy: {accuracy}")
