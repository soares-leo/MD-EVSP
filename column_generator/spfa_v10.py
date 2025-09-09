from initializer.inputs import *
from initializer.utils import calculate_cost
from collections import deque, namedtuple
from datetime import timedelta
import math
import time

# Label carries cumulative stats for one path up to 'node'
class Label:
    __slots__ = [
        'node', 'source_trip', 'total_weight', 'total_dist', 'total_time',
        'dist_since_charge', 'time_since_charge', 'current_time', 'path',
        'total_dh_dist', 'total_dh_time', 'total_travel_dist', 
        'total_travel_time', 'total_wait_time', 'total_charge_time' #MODIFIED
    ]
    
    def __init__(self, node, source_trip, total_weight, total_dist, total_time,
                 dist_since_charge, time_since_charge, current_time, path,
                 total_dh_dist, total_dh_time, total_travel_dist, 
                 total_travel_time, total_wait_time, total_charge_time): #MODIFIED
        self.node = node
        self.source_trip = source_trip
        self.total_weight = total_weight
        self.total_dist = total_dist
        self.total_time = total_time
        self.dist_since_charge = dist_since_charge
        self.time_since_charge = time_since_charge
        self.current_time = current_time
        self.path = path
        self.total_dh_dist = total_dh_dist
        self.total_dh_time = total_dh_time
        self.total_travel_dist = total_travel_dist
        self.total_travel_time = total_travel_time
        self.total_wait_time = total_wait_time
        self.total_charge_time = total_charge_time #MODIFIED
    
    def __eq__(self, other):
        """Required for 'if travel_label not in queue:' check"""
        if not isinstance(other, Label):
            return False
        return (self.node == other.node and 
                self.source_trip == other.source_trip and
                self.path == other.path)

def run_spfa(graph, source_node, D, T_d, dh_df, duals, last_route_number=None, filter_graph=True):
    """
    Optimized SPFA algorithm for finding shortest paths with constraints.
    """
    # Pre-filter graph once
    if filter_graph:
        valid_nodes = {node for node in graph.nodes() 
                      if node.startswith(("l", "c")) or node == source_node}
        graph = graph.subgraph(valid_nodes)
    
    # Cache node attributes for faster access
    node_attrs = {node: graph.nodes[node] for node in graph.nodes()}
    
    # Pre-compute trip nodes
    trip_nodes = {n for n, d in node_attrs.items() if d.get('type') == 'T'}
    
    # Initialize start label
    start = Label(
        node=source_node, source_trip=None, total_weight=0.0,
        total_dist=0.0, total_time=0.0, dist_since_charge=0.0,
        time_since_charge=0.0, current_time=None, path=[source_node],
        total_dh_dist=0.0, total_dh_time=0.0, total_travel_dist=0.0,
        total_travel_time=0.0, total_wait_time=0.0, total_charge_time=0.0 #MODIFIED
    )
    
    best_labels = {}
    queue = deque([start])
    
    print(f"Finding shortest paths for source {source_node}...")
    spfa_start = time.time()
    
    # Main SPFA loop
    while queue:
        u_label = queue.popleft()
        u = u_label.node
        
        #if u_label.current_time is None:
        #    graph = graph.copy()
        #else:
            # if nodes are the source node, recharging nodes or trip nodes with start time on or before the current time:
        #    valid_nodes = [n for n, data in graph.nodes(data=True)
        #                   if n.startswith(("l", "c", "d")) or data.get("start_time") <= u_label.current_time]
        #    graph = graph.subgraph(valid_nodes)

        need_recharge = False
        
        # Process all neighbors
        for v in graph.neighbors(u):
            
            # Skip depot nodes
            if v.startswith("d"):
                continue
            
            # Handle charging station
            if v.startswith("c"):
                if u.startswith(("d", "c")):
                    continue
                elif u.startswith("l"):
                    new_label = _process_charging_station(
                        u, v, u_label, graph[u][v], T_d
                    )
                    if new_label:
                        prev = best_labels.get(v)
                        if prev is None or new_label.total_weight < prev.total_weight:
                            best_labels[v] = new_label
                            queue.append(new_label)
                            need_recharge = False
                    continue
            
            # Handle trip node
            if v.startswith("l"):
                
                # Skip if recharge is needed
                if need_recharge:
                    continue
                
                # Determine source trip
                if len(u_label.path) == 1:
                    first_trip = v
                else:
                    first_trip = u_label.source_trip
                
                # Get edge data
                dh_dist = graph[u][v]["dist"]
                dh_time = graph[u][v]["time"]
                
                # Calculate timing
                if u_label.current_time is None:
                    clock = node_attrs[v]["start_time"] - timedelta(minutes=dh_time)
                else:
                    clock = u_label.current_time
                    if clock + timedelta(minutes=dh_time) > node_attrs[v]["start_time"]:
                        continue
                
                # Check charging station timing constraint
                if u.startswith("c"):
                    if node_attrs[v]["start_time"] < clock + timedelta(minutes=dh_time):
                        continue
                
                # Calculate wait time
                wait_time = (
                    node_attrs[v]["start_time"] 
                    - (clock + timedelta(minutes=dh_time))
                ).total_seconds() / 60.0
                
                # Calculate travel metrics
                travel_time = node_attrs[v]["planned_travel_time"]
                new_total_time = (
                    u_label.total_time 
                    + dh_time 
                    + wait_time 
                    + travel_time
                )
                
                # Check time constraint
                if new_total_time >= T_d:
                    continue
                
                # Calculate distance metrics
                sc = node_attrs[v]["start_cp"]
                ec = node_attrs[v]["end_cp"]
                travel_dist = dh_df.loc[sc, ec]
                new_dist_since_charge = u_label.dist_since_charge + dh_dist + travel_dist
                
                # Check if recharge needed
                if new_dist_since_charge >= D:
                    need_recharge = True
                    continue
                
                weight = graph[u][v]["reduced_cost"]
                
                # Create new label
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
                    total_wait_time=u_label.total_wait_time + wait_time,
                    total_charge_time=u_label.total_charge_time #MODIFIED
                )
                
                # Update best label if better
                prev = best_labels.get(v)
                if prev is None or travel_label.total_weight < prev.total_weight:
                    best_labels[v] = travel_label
                    if travel_label not in queue:
                        queue.append(travel_label)
    
    print("Shortest paths finding process is done.")
    spfa_runtime = time.time() - spfa_start
    print(f"Time spent finding paths: {spfa_runtime} seconds.")
    
    # Build trip routes
    print("Label correcting process started...")
    label_corr_start = time.time()
    
    trip_routes = _build_trip_routes(
        trip_nodes, best_labels, graph, duals
    )
    
    print("Label correcting process is done.")
    label_corr_runtime = time.time() - label_corr_start
    print(f"Time spent correcting labels: {label_corr_runtime} seconds.")
    
    return trip_routes

def _process_charging_station(u, v, u_label, edge_data, T_d):
    """Process transition to charging station."""
    weight_cs = edge_data["reduced_cost"]
    dist_cs = edge_data["dist"]
    time_cs = edge_data["time"]
    
    # Calculate charging time based on distance since last charge
    charge_minutes = (15.6 * ((u_label.dist_since_charge + dist_cs) / 20) / 30) * 60
    
    # Update clock
    new_clock = (u_label.current_time + timedelta(minutes=time_cs + charge_minutes) 
                 if u_label.current_time else None)
    
    # Create charging label
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
        total_wait_time=u_label.total_wait_time + charge_minutes,
        total_charge_time=u_label.total_charge_time + charge_minutes #MODIFIED
    )
    
    # Check time constraint
    if charge_label.total_time >= T_d:
        return None
    
    return charge_label

def _build_trip_routes(trip_nodes, best_labels, graph, duals):
    """Build trip routes from best labels."""
    trip_routes = {}
    
    for trip_id in trip_nodes:
        # Filter labels that have this trip as source
        trip_labels = [label for label in best_labels.values() 
                      if label.path[1] == trip_id if len(label.path) > 1]
        
        best_cols = []
        
        for a in trip_labels:
            a_full_path = a.path + [a.path[0]]
            a_rec_times = sum(1 for x in a.path if x.startswith("c"))
            
            # Skip if last node before depot is charging station
            if a_full_path[-2].startswith("c"):
                continue
            
            # Calculate final deadhead time
            a_dh_time = a.total_dh_time + graph[a.path[0]][a_full_path[-2]]["time"]
            
            # Get last trip and final beta
            a_last_trip = a.path[-1]
            a_beta = graph[a_last_trip][a.path[0]]["reduced_cost"]
            a_final_weight = a.total_weight + a_beta
            
            # Calculate cost
            a_cost = calculate_cost(
                a_rec_times, 
                a.total_travel_time, 
                a_dh_time, 
                a.total_wait_time - a.total_charge_time, #MODIFIED
            )
            
            # Calculate reduced cost
            only_trips = [t for t in a_full_path if t.startswith("l")]
            trip_duals_sum = sum(duals["alpha"][graph.nodes[t]['id']] for t in only_trips)
            a_reduced_cost = a_cost - trip_duals_sum + duals["beta"][a.path[0]]
            
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
                    "final_time": a.current_time,
                    "total_charge_time": a.total_charge_time #MODIFIED
                }
            }
            
            best_cols.append(route)
        
        trip_routes[trip_id] = best_cols
    
    return trip_routes