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






def run_spfa(graph, source_node):
    distances = {source_node: 0}
    distances.update({k: math.inf for k in graph.keys() if k != source_node})
    queue = deque()
    queue.append(source_node)
    while len(queue) > 0:
        u = queue.popleft()
        for v, dv in graph[u].items():
            if distances[u] + dv < distances[v]:
                distances[v] = distances[u] + dv
                if v not in queue:
                    queue.append(v)
        print(distances)
    return distances

# para cada K:
    # para cada nó i \in T:
        # SPFA, ensuring dist and time. 
            # OBS: na regra de dominancia, o nó que cobre maior distância e tem menor custo, prevalece.




