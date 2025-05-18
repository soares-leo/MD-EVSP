from initializer.inputs import *
from initializer.utils import *
from .utils import add_reduced_cost_info
from collections import deque, namedtuple
from datetime import timedelta
import datetime
import json

# a single “state” (label) in the network
Label = namedtuple("Label", [
    "node",             # current node ID
    "cost",             # reduced cost so far (c_p)
    "dist_since_charge",# distance since last charge/depot (d_p)
    "time",             # cumulative travel time since depot
    "current_time",     # actual clock time at arrival
    "path"              # list of nodes visited so far
])

def run_spfa(graph, source, D, T_d, dh_df):
    # initialize label‐lists
    labels = {n: [] for n in graph.nodes}
    
    # initial label from source depot
    init = Label(
        node=source,
        cost=0.0,
        dist_since_charge=0,
        time=0,
        current_time=None,
        path=[source]
    )
    labels[source].append(init)
    
    # FIFO queue of labels to process
    queue = deque([init])
    
    while queue:
        lbl = queue.popleft()
        u = lbl.node
        
        for v in graph.neighbors(u):
            arc_time = graph[u][v]["time"]
            arc_dist = graph[u][v]["dist"]

            if lbl.current_time is None:
                # first hop: back out the travel time so wait = 0
                clock = graph.nodes[v]["start_time"] - timedelta(minutes=arc_time)
            else:
                clock = lbl.current_time

            if u.startswith("c"):
                if graph.nodes[v]["start_time"] < clock + timedelta(minutes=arc_time):
                    continue
            elif not v.startswith("l"):
                continue
            
            # compute wait & next‐trip stats
            wait = (graph.nodes[v]["start_time"] - 
                    (clock + timedelta(minutes=arc_time))).total_seconds() / 60
            v_planned = graph.nodes[v]["planned_travel_time"]
            sc, ec = graph.nodes[v]["start_cp"], graph.nodes[v]["end_cp"]
            v_dist = dh_df.loc[sc, ec]
            v_end_clock = graph.nodes[v]["end_time"]
            
            tot_time = lbl.time + arc_time + wait + v_planned
            tot_dist = lbl.dist_since_charge + arc_dist + v_dist
            
            # 1) time‐window constraint
            if tot_time > T_d:
                continue
            
            # 2) distance‐since‐charge constraint
            if tot_dist > D:
                depot_cols = dh_df.filter(regex=r"^d\d+_\d+$").columns
                best = dh_df.loc[graph.nodes[u]["end_cp"], depot_cols].idxmin()
                cs = "c" + best[1:]
                
                cost_cs = graph[u][cs]["reduced_cost"]
                dist_cs = graph[u][cs]["dist"]
                time_cs = graph[u][cs]["time"]
                charge_minutes = 15.6 * (((lbl.dist_since_charge + dist_cs) / 20) * 60) / 30
                new_clock = clock + timedelta(minutes=time_cs + charge_minutes)
                
                new_lbl = Label(
                    node=cs,
                    cost=lbl.cost + cost_cs,
                    dist_since_charge=0.0,
                    time=lbl.time + time_cs + charge_minutes,
                    current_time=new_clock,
                    path=lbl.path + [cs]
                )
                dest = cs
            else:
                cost_uv = graph[u][v]["reduced_cost"]
                new_lbl = Label(
                    node=v,
                    cost=lbl.cost + cost_uv,
                    dist_since_charge=tot_dist,
                    time=tot_time,
                    current_time=v_end_clock,
                    path=lbl.path + [v]
                )
                dest = v
            
            # Dominance test
            dominated = False
            to_remove = []
            for existing in labels[dest]:
                fixed_ct = existing.current_time or datetime.datetime.min
                if (existing.cost <= new_lbl.cost and
                    existing.dist_since_charge <= new_lbl.dist_since_charge and
                    fixed_ct <= new_lbl.current_time):
                    dominated = True
                    break
                if (new_lbl.cost <= existing.cost and
                    new_lbl.dist_since_charge <= existing.dist_since_charge and
                    new_lbl.current_time <= fixed_ct):
                    to_remove.append(existing)
            if dominated:
                continue
            for ex in to_remove:
                labels[dest].remove(ex)
            
            labels[dest].append(new_lbl)
            queue.append(new_lbl)
    
    # Option B: close every trip‐ending label back to this source depot
    completed_routes = []
    for lbl_list in labels.values():
        for lbl in lbl_list:
            if lbl.node.startswith("l"):
                # add the reduced cost of the final arc i->d
                ret_cost = graph[lbl.node][source]["reduced_cost"]
                full_cost = lbl.cost + ret_cost
                full_path = lbl.path + [source]

                recharging_freq = len(list(filter(lambda x: x[0] == "c", full_path)))
                route_cost = calculate_cost(recharging_freq, travel_time, dh_time, waiting_time)
                
                completed_routes.append(
                    Label(
                        node=source,
                        cost=full_cost,
                        dist_since_charge=lbl.dist_since_charge,
                        time=lbl.time,
                        current_time=lbl.current_time,
                        path=full_path,
                        path_cost=route_cost,
                        path_data= {
                            "total_dh_dist": total_dh_dist,
                            "total_dh_time": total_dh_time,
                            "total_travel_dist": total_travel_dist,
                            "total_travel_time": total_travel_time,
                            "total_wait_time": total_wait_time,
                            "final_time": current_time
                        }
                    )
                )

   
    return completed_routes