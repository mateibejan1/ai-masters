"""
    Problem Solving & Search Assignment 1: Uninformed Search Methods (DFS & BFS)
    Author: Matei Bejan, group 507

    DFS implementation taken from https://www.geeksforgeeks.org/depth-first-search-or-dfs-for-a-graph/.
    BFS implementation taken from https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/.

    The dataset used is "Social circles: Facebook", from https://snap.stanford.edu/data/ego-Facebook.html. 
    Data consists of anonymized 'circles' (or 'friends lists') from Facebook. However, this setup can 
    mimic other networks such as LinkedIn, Twitter etc.

    The problem we're aim to solve is to check whether we can get into contact through a friend of a friend to a 
    certain user of the network. In the case of LinkedIn, assume we are looking for a (new) job. We might want to 
    ask our friend or close connection to refer us to a employee, manager or recruiter which they know, or forward
    our  friend or to ask them to refer us further. We wish to see how many such people can we contact and we can
    plan a contact strategy for each potential future employer.

    Requirements: matplotlib, networkx
"""

from collections import defaultdict, deque
import networkx as nx
import matplotlib.pyplot as plt

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def get_neighbours(self, v):
        return self.graph[v]

    def DFSUtil(self, root, visited):
        visited.append(root)
        for neighbour in self.graph[root]:
            if neighbour not in visited:
                self.DFSUtil(neighbour, visited)

    def DFS(self, root):
        visited = []
        self.DFSUtil(root, visited)
        return visited

    def BFS(self, root):
        visited_flags = [False] * (max(self.graph) + 1)
        queue = []
        queue.append(root)
        visited_flags[root] = True
        visited = []

        while queue:
            root = queue.pop(0)
            visited.append(root)
            for i in self.graph[root]:
                if visited_flags[i] == False:
                    queue.append(i)
                    visited_flags[i] = True
                    
        return visited

graph_visualize = nx.Graph()
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
        if counter < 10:
            graph_visualize.add_edge(from_node, to_node)
            counter += 1
        
        graph_compute.add_edge(from_node, to_node)

root_node = 2

print("DFS starting root {}:".format(root_node))
visited_bfs = graph_compute.BFS(root_node)           
print(visited_bfs[1:])

colour_map = []
for node in graph_visualize:
    if node == root_node:
        colour_map.append('red')
    elif node in visited_bfs:
        colour_map.append('#b2f59f')
    else:
        colour_map.append('#42e3f5')

nx.draw(graph_visualize, with_labels=True, node_size=200, node_color=colour_map)
plt.show()

print("BFS starting root {}:".format(root_node))
visited_dfs = graph_compute.DFS(root_node)
print(visited_dfs[1:])

colour_map = []
for node in graph_visualize:
    if node == root_node:
        colour_map.append('red')
    elif node in visited_dfs:
        colour_map.append('#b2f59f')
    else:
        colour_map.append('#42e3f5')

nx.draw(graph_visualize, with_labels=True, node_size=200, node_color=colour_map)
plt.show()