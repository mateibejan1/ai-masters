"""
    Problem Solving & Search Assignment 3: Local Search Methods (Hill Climbing Search)
    Author: Matei Bejan, group 507

    The dataset used is "Social circles: Facebook", from https://snap.stanford.edu/data/ego-Facebook.html. 
    Data consists of anonymized 'circles' (or 'friends lists') from Facebook. However, this setup can 
    mimic other networks such as LinkedIn, Twitter etc.

    The problem we're aim to solve is to find the least expensive path to get a certain announcement (job posting, 
    workshop advertising, job searching) through the network. In the case of LinkedIn, assume we are looking for a 
    (new) job. We might want to send our inquiry to people who we are closest to, and we want them to send it to 
    their closest connections and so on. The reasoning behind this is to strengthen the power behind the referral 
    someone will receive and multiply our chances of getting hired.
    
    In this scenario connections are represented by numbers from 1 to 10, 1 the strongest connection and 10 being the
    weakest.

    Requirements: matplotlib, networkx, numpy.
"""

import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def randomSolution(tsp):
    cities = list(range(len(tsp)))
    solution = []

    for i in range(len(tsp)):
        randomCity = cities[random.randint(0, len(cities) - 1)]
        solution.append(randomCity)
        cities.remove(randomCity)

    return solution

def routeLength(tsp, solution):
    routeLength = 0
    for i in range(len(solution)):
        routeLength += tsp[solution[i - 1]][solution[i]]
    return routeLength

def getNeighbours(solution):
    neighbours = []
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            neighbour = solution.copy()
            neighbour[i] = solution[j]
            neighbour[j] = solution[i]
            neighbours.append(neighbour)
    return neighbours

def getBestNeighbour(tsp, neighbours):
    bestRouteLength = routeLength(tsp, neighbours[0])
    bestNeighbour = neighbours[0]
    for neighbour in neighbours:
        currentRouteLength = routeLength(tsp, neighbour)
        if currentRouteLength < bestRouteLength:
            bestRouteLength = currentRouteLength
            bestNeighbour = neighbour
    return bestNeighbour, bestRouteLength

def hillClimbing(tsp):
    currentSolution = randomSolution(tsp)
    currentRouteLength = routeLength(tsp, currentSolution)
    neighbours = getNeighbours(currentSolution)
    bestNeighbour, bestNeighbourRouteLength = getBestNeighbour(tsp, neighbours)

    while bestNeighbourRouteLength < currentRouteLength:
        currentSolution = bestNeighbour
        currentRouteLength = bestNeighbourRouteLength
        neighbours = getNeighbours(currentSolution)
        bestNeighbour, bestNeighbourRouteLength = getBestNeighbour(tsp, neighbours)

    return currentSolution, currentRouteLength

maxi = -1
edge_labels = {}
max_nodes = 15

with open("facebook_combined.txt") as f:
    lines = f.readlines()
    counter = 0
    curr_from_node = None
    for line in lines:
        from_node, to_node = int(line.split(' ')[0]), int(line.split(' ')[1])
        if maxi < from_node:
            maxi = from_node
        if maxi < to_node:
            maxi = to_node

graph_compute = np.empty((maxi, maxi), dtype=int)
graph_compute = graph_compute[:max_nodes, :max_nodes]

for i in range(max_nodes):
    for j in range(max_nodes):
        if i == j:
            graph_compute[i, j] = 0
        else:
            graph_compute[i, j] = random.randint(1, 10)
            edge_labels[(i, j)] = graph_compute[i, j]

# print(graph_compute, '\n')

solution = hillClimbing(graph_compute[:max_nodes, :max_nodes])
print(solution)

graph_visualize = nx.Graph()

with open("facebook_combined.txt") as f:
    lines = f.readlines()
    counter = 0
    curr_from_node = None
    all_nodes = set()
    for line in lines:
        from_node, to_node = int(line.split(' ')[0]), int(line.split(' ')[1])
        if from_node in range(1, max_nodes):
            graph_visualize.add_edge(from_node, to_node)
        elif to_node in range(1, max_nodes):
            graph_visualize.add_edge(from_node, to_node)

colour_map = []
for node in graph_visualize:
    if node == solution[0][0]:
        colour_map.append('red')
    elif node in solution[0][1:]:
        colour_map.append('#f59ff2')
    else:
        colour_map.append('#42e3f5')

pos = nx.kamada_kawai_layout(graph_visualize)
nx.draw(graph_visualize, pos, with_labels=True, node_size=200, node_color=colour_map)
plt.show()

