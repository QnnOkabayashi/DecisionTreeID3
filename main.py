from decision_tree import DecisionTree
from dataset import Dataset

percent_training = 0.9

data = Dataset.from_user()
training, testing = data.partition(percent_training)

tree = data.build_tree()

tree.train(training)

print(tree)

if percent_training < 1:

    accuracy = tree.test(testing)

    print(f"Training with {len(training)} cases")
    print(f"Testing with {len(testing)} cases")
    print(f"Tree accuracy: {accuracy}")
