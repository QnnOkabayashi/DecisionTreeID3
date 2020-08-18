# DecisionTreeID3
### By Quinn Okabayashi

An implementation of the ID3 decision tree algorithm in pure Python.

This project is a refactoring of a project I did for a course in highschool. 
The original project can be found here.

# Docs

## Dataset class

The `Dataset` class is used to load datasets, as well as construct `DecisionTree` objects.

### Methods
___
<details>
<summary><code>
def __init__(self, dataset_name: str) -> Dataset
</code></summary>

Creates a new `Dataset` object, where `dataset_name` is the name of a CSV file in `datasets/`, not including the file extension.

Example:
```python
data = Dataset('titanic')
```
</details>

___
<details>
<summary><code>
def from_user(cls) -> Dataset
</code></summary>

Creates a new `Dataset` object by prompting the user to select a dataset from `datasets/`.

Example:
```python
Dataset.from_user()
```
Output:
```
0) breast-cancer
1) primary-tumor
2) titanic
3) congress84
4) tennis
5) mushrooms
Select a dataset: 
```
</details>

___
<details>
<summary><code>
def partition(self, percent_training: float) -> Tuple[List[Case], List[Case]]
</code></summary>

Returns randomly selected training and testing batches, given a percentage of cases to use for training.

Raises `ValueError` if percent_training isn't between 0, exclusive, and 1, inclusive.

Example:
```python
data = Dataset('tennis')
training, testing = data.partition(percent_training=0.9)
```
</details>

___
<details>
<summary><code>
def build_tree(self) -> DecisionTree
</code></summary>

Creates a `DecisionTree` object from the dataset's attributes and attribute fields.

* Note: The cases in the dataset are not exposed to the tree in any way.

Example:
```python
data = Dataset('tennis')
tree = data.build_tree()
```
</details>

___
## DecisionTree class

The `DecisionTree` class is used to create and traverse decision trees.

### Methods
___
<details>
<summary><code>
def __init__(self, names: List[AttrName], opts: List[Set[AttrOpt]]) -> DecisionTree
</code></summary>

Creates a new `DecisionTree` object, where `names` are the attribute names, and `opts` is the corresponding sets of options a case could have for that attribute.

* Note: It is preferable to construct `DecisionTree` objects from the `Dataset` method, `build_tree()`.

Example:
```python
tree = DecisionTree(
    names = [
        'sex', 
        'age_range'
    ],
    opts = [
        {'male', 'female', 'other'}, 
        {'<18', '19-32', '33-65', '>66'}
    ]
)
```
</details>

___
<details>
<summary><code>
def train(self, training_cases: Iterable[Case]) -> None
</code></summary>

Trains the `DecisionTree` object on a batch of training cases.

Example:
```python
data = Dataset('tennis')
training, _ = data.partition(percent_training=0.9)
tree = data.build_tree()
tree.train(training)
```
</details>

___
<details>
<summary><code>
def test(self, testing_cases: Iterable[Case]) -> float
</code></summary>

Returns the % accuracy of the trained `DecisionTree` object on a testing batch. Will return `0` if the tree hasn't been trained.

Example:
```python
data = Dataset('tennis')
training, testing = data.partition(percent_training=0.9)
tree = data.build_tree()
tree.train(training)
accuracy = tree.test(testing)
```
</details>

___
<details>
<summary><code>
def __repr__(self) -> str
</code></summary>

Returns the string representation of the tree.

If the `DecisionTree` object isn't trained, returns `"empty tree"`

Example:
```python
data = Dataset('tennis')
training, testing = data.partition(percent_training=0.9)
tree = data.build_tree()
tree.train(training)
tree_repr = repr(tree)
```
</details>

___
<details>
<summary><code>
def classify(self, case: Case) -> AttrOpt
</code></summary>

Returns the classification of `case` on the `DecisionTree` object.

If the `DecisionTree` object isn't trained, returns `False`

Example:
```python
data = Dataset('tennis')
training, testing = data.partition(percent_training=0.9)
tree = data.build_tree()
tree.train(training)
category = tree.classify(testing[0])
```
</details>