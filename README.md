# ai_decision_tree
#### By Quinn Okabayashi
An implementation of the ID3 decision tree algorithm in pure Python.

# Docs

## Dataset class
The `Dataset` class is used to load datasets, as well as construct `DecisionTree` objects.
### Methods
___
<details>

<summary>

```python
def __init__(self, dataset_name: str) -> Dataset
```
</summary>

Creates a new `Dataset` object, where `dataset_name` is the name of a CSV file in `datasets/`, not including the file extension.


```python
>>> data = Dataset('titanic')
```
</details>

___
### Construct from user input
```python
@classmethod
def from_user(cls) -> Dataset
```
Creates a new `Dataset` object by prompting the user to select a dataset from `datasets/`.

<details>
<summary>Example:</summary>

```python
>>> Dataset.from_user()
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
### `build_tree() -> DecisionTree`
The `build_tree()` method is used to construct a `DecisionTree` object from a dataset:
```python
>>> my_dataset = Dataset('tennis')
>>> my_dataset.build_tree()
```
* Note: only attribute labels and fields are exposed to the tree.
___
### `partition(percent_training: float) -> Tuple[List[Case], List[Case]]`
To partition the dataset into training and testing batches, use the `partition()`  method:

```python
>>> my_dataset = Dataset('tennis')
>>> training, testing = my_dataset.partition(percent_training=0.8)
```
