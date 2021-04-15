"""
    Problem Solving & Search Assignment 2: Informed Search Methods (A* Search)
    Author: Matei Bejan, group 507

    The dataset used is "Social circles: Facebook", from https://snap.stanford.edu/data/ego-Facebook.html. 
    Data consists of anonymized 'circles' (or 'friends lists') from Facebook. However, this setup can 
    mimic other networks such as LinkedIn, Twitter etc.

    The problem we're aim to solve is to check whether we can get into contact through a friend of a friend to a 
    certain user of the network. In the case of LinkedIn, assume we are looking for a (new) job. We might want to 
    ask our friend or close connection to refer us to a employee, manager or recruiter which they know, or forward
    our  friend or to ask them to refer us further. 
    
    We wish to see how many such people can we contact and we can plan a contact strategy for each potential future employer.
    Each person in the connections graph can make one of the following types of referrals:

        1: Referral to a friend.
        2. Referral to a superior.
        3. Referral to an equal in hierarchy.
        4. Referral to an employee.

    Requirements: matplotlib, networkx
"""

import networkx as nx
import matplotlib.pyplot as plt
from random import randint
from collections import OrderedDict 

class Node:
    
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent
        self.g = 0
        
    def __eq__(self, other):
        return self.name == other.name

    def __lt__(self, other):
            return self.g < other.g

    def __repr__(self):
        return ('({0},{1})'.format(self.name, self.g))

class Graph:

    def __init__(self, graph_dict=None, directed=True):
        self.graph_dict = graph_dict or {}
        self.directed = directed
        if not directed:
            self.make_undirected()

    def make_undirected(self):
        for a in list(self.graph_dict.keys()):
            for (b, dist) in self.graph_dict[a].items():
                self.graph_dict.setdefault(b, {})[a] = dist

    def add_edge(self, A, B, distance=1):
        self.graph_dict.setdefault(A, {})[B] = distance
        if not self.directed:
            self.graph_dict.setdefault(B, {})[A] = distance

    def get(self, a, b=None):
        links = self.graph_dict.setdefault(a, {})
        if b is None:
            return links
        else:
            return links.get(b)

    def nodes(self):
        s1 = set([k for k in self.graph_dict.keys()])
        s2 = set([k2 for v in self.graph_dict.values() for k2, v2 in v.items()])
        nodes = s1.union(s2)
        return list(nodes)

def add_to_open_nodes(open_nodes, neighbor):
    for node in open_nodes:
        if (neighbor == node and neighbor.g > node.g):
            return False
    return True

def astar_search(graph, start, end):
    open_nodes = []
    closed_nodes = []

    start_node = Node(start, None)
    goal_node = Node(end, None)

    open_nodes.append(start_node)
    
    while len(open_nodes) > 0:
        open_nodes.sort()
        current_node = open_nodes.pop(0)
        closed_nodes.append(current_node)
        
        if current_node == goal_node:
            path = {}
            while current_node != start_node:
                path[current_node.name] = current_node.g
                current_node = current_node.parent
            path[current_node.name] = current_node.g
            return OrderedDict(sorted(path.items()))

        neighbors = graph.get(current_node.name)

        for key, value in neighbors.items():
            neighbor = Node(key, current_node)

            if(neighbor in closed_nodes):
                continue

            neighbor.g = current_node.g + graph.get(current_node.name, neighbor.name)

            if(add_to_open_nodes(open_nodes, neighbor) == True):
                open_nodes.append(neighbor)

    return None

graph_compute = Graph()

with open("facebook_combined.txt") as f:
    lines = f.readlines()
    counter = 0
    curr_from_node = None
    all_nodes = set()
    for line in lines:
        from_node, to_node = int(line.split(' ')[0]), int(line.split(' ')[1])
        if from_node < 25 and from_node > len(all_nodes):
            counter = 0
            all_nodes.add(from_node)
        
        graph_compute.add_edge(from_node, to_node, randint(1, 4))

root_node, destination_node = 2, 279
path = astar_search(graph_compute, root_node, destination_node)
print("Cummulative prices:", path)

graph_visualize = nx.Graph()

with open("facebook_combined.txt") as f:
    lines = f.readlines()
    counter = 0
    curr_from_node = None
    all_nodes = set()
    for line in lines:
        from_node, to_node = int(line.split(' ')[0]), int(line.split(' ')[1])
        if from_node in list(path.keys()):
            graph_visualize.add_edge(from_node, to_node)
        elif to_node in list(path.keys()):
            graph_visualize.add_edge(from_node, to_node)

individual_prices_proxy = list(path.items())
individual_prices = []

i = len(individual_prices_proxy) - 1
while i > 0:
    individual_prices.append((individual_prices_proxy[i][0], individual_prices_proxy[i][1] - individual_prices_proxy[i - 1][1]))
    i -= 1
individual_prices.append(individual_prices_proxy[0])
individual_prices = dict(list(reversed(individual_prices)))
print("Individual prices:", OrderedDict((sorted(individual_prices.items()))))

colour_map = []
for node in graph_visualize:
    if node == root_node:
        colour_map.append('red')
    if node in list(individual_prices.keys()):
        if individual_prices[node] == 1:
            colour_map.append('#b2f59f') #green
        elif individual_prices[node] == 2: 
            colour_map.append('#eff59f') #yellow
        elif individual_prices[node] == 3:
            colour_map.append('#a29ff5') #blue
        elif individual_prices[node] == 4:
            colour_map.append('#f59ff2') #pink
    else:
        colour_map.append('#42e3f5')

nx.draw(graph_visualize, with_labels=True, node_size=200, node_color=colour_map)
plt.show()