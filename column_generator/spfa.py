from initializer.inputs import *
from initializer.utils import calculate_cost  # ensure calculate_cost is imported
from collections import deque, namedtuple
from datetime import timedelta
import datetime

# a single “state” (label) in the network, now with accumulators
Label = namedtuple("Label", [
    "node",
    "cost",
    "dist_since_charge",
    "time",
    "current_time",
    "path",
    "total_dh_dist",
    "total_dh_time",
    "total_travel_dist",
    "total_travel_time",
    "total_wait_time"
])

def run_spfa(graph, source, D, T_d, dh_df, last_route_number, duals):
    # initialize label‐lists
    labels = {n: [] for n in graph.nodes}
    
    # initial label from source depot
    init = Label(
        node=source,
        cost=0.0,
        dist_since_charge=0,
        time=0,
        current_time=None,
        path=[source],
        total_dh_dist=0.0,
        total_dh_time=0.0,
        total_travel_dist=0.0,
        total_travel_time=0.0,
        total_wait_time=0.0
    )
    labels[source].append(init)
    
    # FIFO queue
    queue = deque([init])
    
    while queue:
        lbl = queue.popleft()
        u = lbl.node
        
        for v in graph.neighbors(u):
            arc_time = graph[u][v]["time"]
            arc_dist = graph[u][v]["dist"]

            # first hop clock adjustment
            if lbl.current_time is None:
                clock = graph.nodes[v]["start_time"] - timedelta(minutes=arc_time)
            else:
                clock = lbl.current_time

            # time‐compatibility
            if u.startswith("c"):
                if graph.nodes[v]["start_time"] < clock + timedelta(minutes=arc_time):
                    continue
            elif not v.startswith("l"):
                continue
            
            # compute wait, trip stats
            wait = (graph.nodes[v]["start_time"] - 
                    (clock + timedelta(minutes=arc_time))).total_seconds() / 60
            v_planned = graph.nodes[v]["planned_travel_time"]
            sc, ec = graph.nodes[v]["start_cp"], graph.nodes[v]["end_cp"]
            v_dist = dh_df.loc[sc, ec]
            v_end_clock = graph.nodes[v]["end_time"]
            
            # new totals
            tot_time = lbl.time + arc_time + wait + v_planned
            tot_dist = lbl.dist_since_charge + arc_dist + v_dist
            
            # constraints
            if tot_time > T_d:
                continue
            
            # branch on recharge or trip
            if tot_dist > D:
                # recharge node
                depot_cols = dh_df.filter(regex=r"^d\d+_\d+$").columns
                best = dh_df.loc[graph.nodes[u]["end_cp"], depot_cols].idxmin()
                cs = "c" + best[1:]
                
                cost_cs = graph[u][cs]["reduced_cost"]
                dist_cs = graph[u][cs]["dist"]
                time_cs = graph[u][cs]["time"]
                charge_minutes = (15.6 * ((lbl.dist_since_charge + dist_cs) / 20) / 30) * 60
                new_clock = clock + timedelta(minutes=time_cs + charge_minutes)
                
                new_lbl = Label(
                    node=cs,
                    cost=lbl.cost + cost_cs,
                    dist_since_charge=0.0,
                    time=lbl.time + time_cs + charge_minutes,
                    current_time=new_clock,
                    path=lbl.path + [cs],
                    total_dh_dist=lbl.total_dh_dist + dist_cs,
                    total_dh_time=lbl.total_dh_time + time_cs,
                    total_travel_dist=lbl.total_travel_dist,
                    total_travel_time=lbl.total_travel_time,
                    total_wait_time=lbl.total_wait_time
                )

                if new_lbl.time > T_d:
                    continue

                dest = cs
            else:
                # trip extension
                cost_uv = graph[u][v]["reduced_cost"]
                new_lbl = Label(
                    node=v,
                    cost=lbl.cost + cost_uv,
                    dist_since_charge=tot_dist,
                    time=tot_time,
                    current_time=v_end_clock,
                    path=lbl.path + [v],
                    total_dh_dist=lbl.total_dh_dist + arc_dist,
                    total_dh_time=lbl.total_dh_time + arc_time,
                    total_travel_dist=lbl.total_travel_dist + v_dist,
                    total_travel_time=lbl.total_travel_time + v_planned,
                    total_wait_time=lbl.total_wait_time + wait
                )
                dest = v
            
            # Dominance prune
            dominated = False
            to_remove = []
            for existing in labels[dest]:
                #fixed_ct = existing.current_time or datetime.datetime.min
                if (existing.cost <= new_lbl.cost and
                    existing.dist_since_charge >= new_lbl.dist_since_charge): #and
                    #fixed_ct <= new_lbl.current_time):
                    dominated = True
                    break
                if (new_lbl.cost <= existing.cost and
                    new_lbl.dist_since_charge >= existing.dist_since_charge): #and
                    #new_lbl.current_time <= fixed_ct):
                    to_remove.append(existing)
            if dominated:
                continue
            for ex in to_remove:
                labels[dest].remove(ex)
            
            labels[dest].append(new_lbl)
            queue.append(new_lbl)
    
    completed_routes = {}
    route_idx = last_route_number
    for lbl_list in labels.values():
        for lbl in lbl_list:
            if lbl.node.startswith("l"):
                # final arc cost back to depot
                ret_cost = graph[lbl.node][source]["reduced_cost"]
                full_cost = lbl.cost + ret_cost
                full_path = lbl.path + [source]
                
                # calculate final route cost
                recharging_freq = sum(1 for x in full_path if x.startswith("c"))
                route_cost = calculate_cost(
                    recharging_freq,
                    lbl.total_travel_time,
                    lbl.total_dh_time,
                    lbl.total_wait_time
                )
                
                
                ### NOVA IMPLEMENTACAO
                trips_in_p = [trip for trip in full_path if trip.startswith("l")]
                dep = full_path[0]
                true_reduced_cost = route_cost - sum(duals["alpha"][trip] for trip in trips_in_p) + duals['beta'][dep]
                ### FIM DA NOVA IMPLEMENTACAO
                
                # assemble entry
                completed_routes[f"Route_{route_idx}"] = {
                    "Path": full_path,
                    "Cost": route_cost,
                    "ReducedCost": true_reduced_cost,
                    "Data": {
                        "total_dh_dist": lbl.total_dh_dist,
                        "total_dh_time": lbl.total_dh_time,
                        "total_travel_dist": lbl.total_travel_dist,
                        "total_travel_time": lbl.total_travel_time,
                        "total_wait_time": lbl.total_wait_time,
                        "final_time": lbl.current_time
                    }
                }
                route_idx += 1

    return completed_routes

# WHAT NEEDS TO BE DONE HERE IS TO ALL GENERATED ROUTES STARTING FROM A GIVEN TRIP I AND APPLY THE DOMINANCE RULES AFTER ALL ROUTES WERE GENERATED.