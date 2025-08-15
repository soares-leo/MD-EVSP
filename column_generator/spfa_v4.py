from initializer.inputs import *
from initializer.utils import calculate_cost
from collections import deque, namedtuple, defaultdict
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

def run_spfa(graph, source_node, D, T_d, dh_df, duals, last_route_number=None):
    # Initialize the start label
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
        total_wait_time=0.0
    )

    # Best-known (non-dominated) labels per node. Maps node -> list of Labels
    best_labels = defaultdict(list)
    # Queue of labels to relax
    queue = deque([start])

    print("Finding shortest paths for source", source_node, "...")
    spfa_start_time = time.time()
    
    while queue:
        u_label = queue.popleft()
        u = u_label.node

        # Explore all possible next nodes
        for v in graph.neighbors(u):
            if not v.startswith("l"):
                continue

            dh_dist = graph[u][v]["dist"]
            dh_time = graph[u][v]["time"]

            # determine clock at departure from u
            if u_label.current_time is None:
                # first move: align so that arrival at v matches its start_time
                clock = graph.nodes[v]["start_time"] - timedelta(minutes=dh_time)
            else:
                clock = u_label.current_time

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
            
            # --- START: New Dominance Logic ---
            def process_label(new_label):
                """Processes a new label using dominance rules."""
                node_to_update = new_label.node
                
                # 1. Check if the new label is dominated by any existing label for this node.
                # A label is dominated if another is better or equal on all key metrics.
                is_dominated = False
                for existing_label in best_labels[node_to_update]:
                    if (existing_label.total_weight <= new_label.total_weight and
                        existing_label.total_time <= new_label.total_time and
                        existing_label.dist_since_charge <= new_label.dist_since_charge):
                        # To be a true dominance, at least one must be strictly better.
                        # This prevents identical labels from dominating each other.
                        if (existing_label.total_weight < new_label.total_weight or
                            existing_label.total_time < new_label.total_time or
                            existing_label.dist_since_charge < new_label.dist_since_charge):
                            is_dominated = True
                            break
                
                if is_dominated:
                    return # Do not add the new label, it's suboptimal.

                # 2. Remove any existing labels that are now dominated by the new label.
                non_dominated_list = []
                for existing_label in best_labels[node_to_update]:
                    is_dominated_by_new = (new_label.total_weight <= existing_label.total_weight and
                                           new_label.total_time <= existing_label.total_time and
                                           new_label.dist_since_charge <= existing_label.dist_since_charge)
                    
                    if not is_dominated_by_new:
                        non_dominated_list.append(existing_label)
                
                # 3. Add the new, non-dominated label and add it to the queue.
                non_dominated_list.append(new_label)
                best_labels[node_to_update] = non_dominated_list
                queue.append(new_label)
            # --- END: New Dominance Logic ---
            
            if new_dist_since_charge >= D:
                # need an en route charge first
                depot_cols = dh_df.filter(regex=r"^d\d+_\d+$").columns
                best = dh_df.loc[graph.nodes[u]["end_cp"], depot_cols].idxmin()
                cs = "c" + best[1:]
                dist_cs = graph[u][cs]["dist"]
                time_cs = graph[u][cs]["time"]
                charge_minutes = (15.6 * ((u_label.dist_since_charge + dist_cs) / 20) / 30) * 60
                new_clock = clock + timedelta(minutes=time_cs + charge_minutes)

                if u_label.total_time + time_cs + charge_minutes >= T_d:
                    continue

                weight_cs = graph[u][cs]["reduced_cost"]
                charge_label = Label(
                    node=cs,
                    source_trip=u_label.source_trip,
                    total_weight=u_label.total_weight + weight_cs,
                    total_dist=u_label.total_dist + dist_cs,
                    total_time=u_label.total_time + time_cs + charge_minutes,
                    dist_since_charge=0.0,
                    time_since_charge=0.0,
                    current_time=new_clock,
                    path=u_label.path + [cs],
                    total_dh_dist=u_label.total_dh_dist + dist_cs,
                    total_dh_time=u_label.total_dh_time + time_cs,
                    total_travel_dist=u_label.total_travel_dist,
                    total_travel_time=u_label.total_travel_time,
                    total_wait_time=u_label.total_wait_time + charge_minutes
                )
                
                process_label(charge_label) # Use the new dominance-based processing
                continue

            # now the actual travel leg to v
            weight = graph[u][v]["reduced_cost"]
            first_trip = v if len(u_label.path) == 1 else u_label.source_trip

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
            
            process_label(travel_label) # Use the new dominance-based processing

    print("Shortest paths finding process is done.")
    spfa_end_time = time.time()
    spfa_runtime = spfa_end_time - spfa_start_time
    print("Time spent finding paths:", spfa_runtime, "seconds.")
    
    print("Label correcting process started...")
    label_corr_start_time = time.time()

    T = [n for n, d in graph.nodes(data=True) if d.get('type') == 'T']
    trip_routes = {}
    
    # Flatten the list of lists of labels into a single list of all final path labels
    all_final_labels = [label for sublist in best_labels.values() for label in sublist]

    for trip_id in T:
        # Filter all generated paths to find those that start with the current trip_id
        candidate_labels = list(filter(lambda x: len(x.path) > 1 and x.path[1] == trip_id, all_final_labels))
        
        best_cols = []
        for a in candidate_labels:
            a_full_path = a.path + [a.path[0]]
            a_rec_times = sum([1 if x.startswith("c") else 0 for x in a.path])
            if a_full_path[-2].startswith("c"):
                continue
            
            # Use graph data for the final leg back to the depot
            last_node_in_path = a_full_path[-2]
            depot_node = a_full_path[-1]
            final_leg_dh_time = graph[last_node_in_path][depot_node]["time"]
            final_leg_reduced_cost = graph[last_node_in_path][depot_node]["reduced_cost"]

            a_dh_time = a.total_dh_time + final_leg_dh_time
            a_final_weight = a.total_weight + final_leg_reduced_cost
            a_cost = calculate_cost(a_rec_times, a.total_travel_time, a_dh_time, a.total_wait_time)
            only_trips = [t for t in a_full_path if t.startswith("l")]
            a_reduced_cost = a_cost - sum([duals["alpha"][graph.nodes[t]['id']] for t in only_trips]) + duals["beta"][a.path[0]]

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

        trip_routes[trip_id] = best_cols
    
    print("Label correcting process is done.")
    label_corr_end_time = time.time()
    label_corr_runtime = label_corr_end_time - label_corr_start_time
    print("Time spent correcting labels:", label_corr_runtime, "seconds.")

    return trip_routes