import math
from collections import deque

graph = {
    "A": {"B": 2, "F": 4},
    "B": {"C": 3, "D": 3, "F": 6},
    "C": {},
    "D": {"E": 2, "F": 2},
    "E": {"G": 3},
    "F": {"E": 1},
    "G": {}
}

graph_2 = {
    1: {2: 1, 3: 8, 4: 9},
    2: {3: 7, 4: -2, 5: 3},
    3: {4: 3},
    4: {5: -3},
    5: {}
}

# def run_spfa(graph, source_node):
#     distances = {source_node: 0}
#     distances.update({k: math.inf for k in graph.keys() if k != source_node})
#     queue = deque()
#     queue.append(source_node)
#     while len(queue) > 0:
#         u = queue.popleft()
#         for v, dv in graph[u].items():
#             if distances[u] + dv < distances[v]:
#                 distances[v] = distances[u] + dv
#                 if v not in queue:
#                     queue.append(v)
#         print(distances)
#     return distances

# shortest_path = run_spfa(graph, "A")
# print(shortest_path)






from collections import deque
import math

def run_spfa(graph, source_node):
    # 1. Initialize
    distances   = {node: math.inf for node in graph}
    predecessor = {node: None       for node in graph}
    distances[source_node] = 0

    queue = deque([source_node])

    # 2. Main SPFA loop
    while queue:
        u = queue.popleft()
        for v, weight in graph[u].items():
            new_dist = distances[u] + weight
            if new_dist < distances[v]:
                distances[v] = new_dist
                predecessor[v] = u
                if v not in queue:
                    queue.append(v)

    # 3. Return both distance and tree info
    return distances, predecessor

def reconstruct_path(predecessor, source, target):
    """Walk backwards from target to source using predecessor[]"""
    path = []
    node = target
    while node is not None:
        path.append(node)
        node = predecessor[node]
    path.reverse()
    # if it doesn’t start with source, there was no path
    if path and path[0] == source:
        return path
    return []  # unreachable


distances, pred = run_spfa(graph, 'A')
print(distances)  
# {'A': 0, 'B': 5, 'C': 2, 'D': 6}

print(reconstruct_path(pred, 'A', 'G'))  

# para cada K:
    # para cada nó i \in T:
        # SPFA, ensuring dist and time. 
            # OBS: na regra de dominancia, o nó que cobre maior distância e tem menor custo, prevalece.




