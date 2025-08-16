from initializer.inputs import *
from initializer.utils import calculate_cost  # if you need it elsewhere
from collections import deque, namedtuple
from datetime import timedelta
import math
import time

# A label carries ALL of the cumulative stats for one path up to 'node'
Label = namedtuple("Label", [
    "node",
    "source_trip",
    "total_weight",
    "total_dist",
    "total_time",
    "dist_since_charge",
    "time_since_charge",
    "current_time",
    "path",
    "total_dh_dist",
    "total_dh_time",
    "total_travel_dist",
    "total_travel_time",
    "total_wait_time"
])

# [node for node, attr in G.nodes(data=True) if attr['type'] == 'user']

def run_spfa(graph, source_node, D, T_d, dh_df, duals, last_route_number=None, filter_graph=True):
    # Initialize the start label
    if filter_graph:
        mask = [node for node in graph.nodes() if node.startswith("l") or node.startswith("c") or node == source_node]
        graph = graph.subgraph(mask)

    start = Label(
        node=source_node,
        source_trip=None,
        total_weight=0.0,
        total_dist=0.0,
        total_time=0.0,
        dist_since_charge=0.0,
        time_since_charge=0.0,
        current_time=None,
        path=[source_node],
        total_dh_dist=0.0,
        total_dh_time=0.0,
        total_travel_dist=0.0,
        total_travel_time=0.0,
        total_wait_time=0.0,
    )

    # Best-known (lowest-cost) label per node
    best_labels = {}
    # Predecessor mapping for reconstructing paths
    #predecessor = {}
    # Queue of labels to relax
    queue = deque([start])

    print("Finding shortest paths for source", source_node, "...")
    spfa_start_time = time.time()
    
    while queue:
        u_label = queue.popleft()
        u = u_label.node

        need_recharge = False

        for v in graph.neighbors(u):
            
            if v.startswith("d"):
                continue
            
            if v.startswith("c"):
                if u.startswith("d") or u.startswith("c"):
                    continue
                elif u.startswith("l"):
                    weight_cs = graph[u][v]["reduced_cost"]
                    dist_cs = graph[u][v]["dist"]
                    time_cs = graph[u][v]["time"]
                    charge_minutes = (15.6 * ((u_label.dist_since_charge + dist_cs) / 20) / 30) * 60
                    new_clock = u_label.current_time + timedelta(minutes=time_cs + charge_minutes)

                    charge_label = Label(
                        node=v,
                        source_trip=u_label.source_trip,
                        total_weight=u_label.total_weight + weight_cs,
                        total_dist=u_label.total_dist + dist_cs,
                        total_time=u_label.total_time + time_cs + charge_minutes,
                        dist_since_charge=0.0,
                        time_since_charge=0.0,
                        current_time=new_clock,
                        path=u_label.path + [v],
                        total_dh_dist=u_label.total_dh_dist + dist_cs,
                        total_dh_time=u_label.total_dh_time + time_cs,
                        total_travel_dist=u_label.total_travel_dist,
                        total_travel_time=u_label.total_travel_time,
                        total_wait_time=u_label.total_wait_time + charge_minutes
                    )

                    if charge_label.total_time >= T_d:
                        continue

                    prev = best_labels.get(v)
                        
                    # The way that it is, even if u needs recharge, if the dominance is false it will not be recharged and will probably be left without recharge until the end. This part might need a review.
                    # The way taht it is, it takes into account even the nodes that do not need recharge, as expected given that we are dealing with and adjacency matrix. However, this part might also need a review because of that.
                    if prev is None or charge_label.total_weight < prev.total_weight:
                        best_labels[v] = charge_label
                        queue.append(charge_label)
                        need_recharge = False
                    
                    continue
            
            if v.startswith("l"):
                
                if need_recharge:
                    continue

                weight = graph[u][v]["reduced_cost"]
                if len(u_label.path) == 1:
                    first_trip = v
                else:
                    first_trip = u_label.source_trip

            dh_dist = graph[u][v]["dist"]
            dh_time = graph[u][v]["time"]

            if u_label.current_time is None:
                clock = graph.nodes[v]["start_time"] - timedelta(minutes=dh_time)
            else:
                clock = u_label.current_time
                if clock + timedelta(minutes=dh_time) > graph.nodes[v]["start_time"]:
                    continue

            # if we're coming from a charging node, we must not violate its timing
            if u.startswith("c"):
                if graph.nodes[v]["start_time"] < clock + timedelta(minutes=dh_time):
                    continue

            wait_time = (
                graph.nodes[v]["start_time"]
                - (clock + timedelta(minutes=dh_time))
            ).total_seconds() / 60.0

            travel_time = graph.nodes[v]["planned_travel_time"]
            new_total_time = (
                u_label.total_time
                + dh_time
                + wait_time
                + travel_time
            )

            if new_total_time >= T_d:
                continue

            sc = graph.nodes[v]["start_cp"]
            ec = graph.nodes[v]["end_cp"]
            travel_dist = dh_df.loc[sc, ec]
            new_dist_since_charge = u_label.dist_since_charge + dh_dist + travel_dist
            
            if new_dist_since_charge >= D:
                need_recharge = True
                continue

            weight = graph[u][v]["reduced_cost"]

            travel_label = Label(
                node=v,
                source_trip=first_trip,
                total_weight=u_label.total_weight + weight,
                total_dist=u_label.total_dist + dh_dist + travel_dist,
                total_time=new_total_time,
                dist_since_charge=new_dist_since_charge,
                time_since_charge=u_label.time_since_charge + dh_time + travel_time,
                current_time=clock + timedelta(minutes=dh_time + wait_time + travel_time),
                path=u_label.path + [v],
                total_dh_dist=u_label.total_dh_dist + dh_dist,
                total_dh_time=u_label.total_dh_time + dh_time,
                total_travel_dist=u_label.total_travel_dist + travel_dist,
                total_travel_time=u_label.total_travel_time + travel_time,
                total_wait_time=u_label.total_wait_time + wait_time
            )

            prev = best_labels.get(v)
            if prev is None or travel_label.total_weight < prev.total_weight:
                best_labels[v] = travel_label
                #predecessor[v] = u
                if travel_label not in queue:
                    queue.append(travel_label)

    print("Shortest paths finding process is done.")
    spfa_end_time = time.time()
    spfa_runtime = spfa_end_time - spfa_start_time
    print("Time spent findind paths:", spfa_runtime, "seconds.")
    
    print("Label correcting process started...")
    label_corr_start_time = time.time()

    T = [n for n, d in graph.nodes(data=True) if d.get('type') == 'T']
    trip_routes = {}
    _i = 1

    for trip_id in T:
        trip_routes[trip_id] = list(filter(lambda x: x.path[1] == trip_id, best_labels.values()))
        # for k in trip_routes[trip_id]:
            # print(k.path)
            # print()
        
        best_cols = []
        best_weight = math.inf
        
        for a in trip_routes[trip_id]:
            a_full_path = a.path + [a.path[0]]
            a_rec_times = sum([1 if x.startswith("c") else 0 for x in a.path])
            if a_full_path[-2].startswith("c"):
                continue
            a_dh_time = a.total_dh_time + graph[a.path[0]][a_full_path[-2]]["time"]
            a_last_trip = a.path[-1]
            a_beta = graph[a_last_trip][a.path[0]]["reduced_cost"]
            a_final_weight = a.total_weight + a_beta
            a_cost = calculate_cost(a_rec_times, a.total_travel_time, a_dh_time, a.total_wait_time)
            only_trips = [t for t in a_full_path if t.startswith("l")]
            a_reduced_cost = a_cost - sum([duals["alpha"][graph.nodes[t]['id']] for t in only_trips]) + duals["beta"][a.path[0]]
            
            # if a_final_weight > best_weight:
            #     continue
            
            route = {
                "Path": a_full_path,
                "Cost": a_cost,
                "ReducedCost": a_reduced_cost,
                "Data": {
                    "total_dh_dist": a.total_dh_dist,
                    "total_dh_time": a.total_dh_time,
                    "total_travel_dist": a.total_travel_dist,
                    "total_travel_time": a.total_travel_time,
                    "total_wait_time": a.total_wait_time,
                    "final_time": a.current_time
                }
            }

            best_cols.append(route)
            # if a_final_weight == best_weight:
            #     best_cols.append(route)
            # else:
            #     best_weight = a_final_weight
            #     best_cols = [route]               

        trip_routes[trip_id] = best_cols
    
    print("Label correcting process is done.")
    label_corr_end_time = time.time()
    label_corr_runtime = label_corr_end_time - label_corr_start_time
    print("Time spent correcting labels:", label_corr_runtime, "seconds.")

    # build output distances dict
    # for k, v in best_labels.items():
    #     print(k, ":", v)

    # for k, v in trip_routes.items():
    #     print(k, v)
    #     print()
    # print(best_labels["l4_cp1_1"])
    # print()
    # print(best_labels["l4_cp1_145"]["source_trip"]["l4_cp1_1"])
    # print()
    # print(best_labels["l4_cp1_2"])
    # print()
    # print(best_labels["l4_cp1_137"])
    # print()
    # print(best_labels["l59_cp1_2"])
    # print()
    # print(best_labels["d1_4"])
    return trip_routes
