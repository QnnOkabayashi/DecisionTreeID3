"""
Program by Quinn Okabayashi
Credits:
Find mode of list: goo.gl/dowz5T
Determine if list contains identical items: goo.gl/U3PsbK
Accessing index of loop: goo.gl/AKSkXk
"""

from math import log
from time import time
from random import shuffle
cases = []  # List of all cases
attributes = []  # Holds attribute names so they can be accessed by index for printing. An attribute is a header
statuses = []  # A list of attribute statuses, with the index representing the attribute.

files = ['datasets/tennis.txt',
         'datasets/titanic.txt',
         'datasets/breast-cancer.txt',
         'datasets/congress84.txt',
         'datasets/primary-tumor.txt',
         'datasets/mushrooms.txt']

dataset = files[1]  # Choose which dataset to analyze
train = 0.1  # Choose what percantage of cases to use as training
show = True  # Choose whether to display the decision tree or not


class Node:  # Each node is an object on the tree
    def __init__(self, l, s):
        self.label = l  # Either a finalized category or an attribute name
        self.status = s  # Holds the status of the parent node. Ex. 'sunny'
        self.children = {}  # Key: status, value: child node with that attribute

    def leaf(self):
        return len(self.children) == 0


def read(myfile):
    f = open(myfile, 'r')

    recent = []
    for word in f.readline().strip().split(','):  # Loops through only headers
        attributes.append(word)
        statuses.append([])
        recent.append('')

    for l, line in enumerate(f):  # Loops through each distinct case.
        cases.append([])
        for i, word in enumerate(line.strip().split(',')):
            if word != recent[i]:
                if word not in statuses[i]:  # Dont check to see if status is known, see if it's the same as the last
                    statuses[i].append(word)
                    recent[i] = word
            cases[l].append(word)


def entropy(s):
    atc = []  # atc = attribute count. Index: category, Value: # of occurences
    for stat in statuses[0]:
        n = len([0 for c in s if c[0] == stat])
        if n != 0:
            atc.append(n)

    return -sum((atc[i]*1.0/len(s)) * log(atc[i]*1.0/len(s), 2) for i in range(len(atc)))


def gain(s, a):
    stc = []  # stc = status cases. Index: the status # of an attribute, Value: list of days
    for i, stat in enumerate(statuses[a]):
        n = [c for c in s if c[a] == stat]
        stc.append([])
        if len(n) != 0:
            stc[i] = n

    return (entropy(s) - sum(((len(stc[i]))*1.0/len(s)) * entropy(stc[i]) for i in range(len(stc)))), stc  # Returns gain and attribute dictionary for id3 program


def mode(s):  # Takes a list of cases, returns the most common category
    v = [x[0] for x in s][:]
    return max(set(v), key=v.count)


def id3(s, a, ps):  # s = List of cases, a = List of attributes, ps = Previous status
    if len(set([s[x][0] for x in range(len(s))])) <= 1:  # All same category
        return Node(s[0][0], ps)
    if len(a) == 0:  # No more attributes
        return Node(mode(s), ps)

    bat = ('', -1)  # bat = best attribute. Tuple that holds best attribute and it's gain
    stc = []  # stc = status cases.
    for att in a:
        g = gain(s, att)[:]  # g holds gain so I only have to run gain once per attribute
        if bat[1] < g[0]:
            bat = (att, g[0])
            stc = g[1]

    nl = Node(bat[0], ps)  # nl = non-leaf. Node that will be given children later
    for i, stat in enumerate(statuses[bat[0]]):  # Loop through statuses of bat
        if len(stc[i]) != 0:  # If len(stc[i]) is 0, then there are no days with that attribute
            x = list(a)
            x.remove(bat[0])
            child = id3(stc[i], x, stat)
        else:
            child = Node(mode(s), stat)  # the child is a leaf labeled with the most common category
        nl.children[stat] = child
    return nl


def printer(n, l):  # Recursive function that prints out a tree
    if n.leaf():
        print l*'|  ' + '> ' + n.label
    else:
        print l*'|  ' + attributes[n.label] + '?'
        for c in n.children.values():
            print (l+1)*'|  ' + '[' + c.status + ']'
            printer(c, l+2)


def climb(node, case, depth):  # Recursive function to determine if a case is categorized correctly. "Climbs" the tree
    if node.leaf():  # If node is a leaf, return whether the case has been categorized correctly
        return case[0] == node.label
    return climb(node.children[case[node.label]], case, depth + 1)  # Run again, but with the next node down the tree


def main(f, n, p):  # f = file name, n = % training, p = print?
    if (n > 0) == (n < 1):
        start = time()
        read(f)
        if int(len(cases) * n) == 0:
            raise Exception(str(n * 100) + "% training results in 0 training cases. Use a larger % training value.")
        else:
            shuffle(cases)  # Use random order
            training = cases[:int(len(cases) * n)]  # Partition
            testing = cases[int(len(cases) * n):]
            tree = id3(training, range(len(attributes))[1:], None)  # Create decision tree with ID3 algorithm
            acc = len([0 for c in testing if climb(tree, c, 0)])*1.0/len(testing)  # Finds accuracy of tree
            period = time() - start

            print '\nDataset: ' + f.split('/')[1].rstrip('.txt') + '\n'*2 + 'Training with ' + str(len(training)) + ' cases\nTesting with ' + str(len(testing)) + ' cases\nTime: ' + str(period) + 's\nAccuracy: ' + str(acc) + '\n'
            if p:
                printer(tree, 0)
    elif n == 1:
        start = time()
        read(f)  # Have to run read() first to fill cases[]
        tree = id3(cases, range(len(attributes))[1:], None)  # Create decision tree with ID3 algorithm
        print '\nDataset: ' + f.split('/')[1].rstrip('.txt') + '\n' * 2 + 'Time: ' + str(time() - start) + 's\n'
        if p:
            printer(tree, 0)
    else:
        raise Exception(str(n) + ' Must be in range 0 < n <= 1.')


main(dataset, train, show)
