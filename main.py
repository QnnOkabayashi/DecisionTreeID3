from decision_tree import DecisionTree
from csv_reader import CSVReader
from datasets import Dataset

dataset = Dataset.TITANIC
percent_training = 0.5

data = CSVReader(dataset)
training, testing = data.partition(percent_training)

tree = DecisionTree(data.names, data.opts)
tree.train(training)
accuracy = tree.test(testing)

tree.printer(node=tree.root)

print(f"Tree accuracy: {accuracy}")
