from collections import deque, defaultdict
from datetime import timedelta
import time
from initializer.utils import calculate_cost
from initializer.inputs import problem_params

# --- Index Constants for Label Tuple ---
NODE_IDX = 0
TOTAL_WEIGHT_IDX = 1  # Reduced Cost. Lower is better.
TOTAL_DIST_IDX = 2
TOTAL_TIME_IDX = 3
DIST_SINCE_CHARGE_IDX = 4
TIME_SINCE_CHARGE_IDX = 5
CURRENT_TIME_IDX = 6
PATH_IDX = 7
TOTAL_DH_DIST_IDX = 8
TOTAL_DH_TIME_IDX = 9
TOTAL_TRAVEL_DIST_IDX = 10 # Travel Distance. Higher is better.
TOTAL_TRAVEL_TIME_IDX = 11
TOTAL_WAIT_TIME_IDX = 12
TOTAL_CHARGING_TIME_IDX = 13


def run_spfa(graph, source_node, D, T_d, dh_df, duals):
    """
    Solves the pricing problem using a fast, single-criterion SPFA, followed by
    a "group by first trip" heuristic to select a diverse set of high-quality columns.
    """
    start_label = (
        source_node, 0.0, 0.0, 0.0, 0.0, 0.0, None, (source_node,),
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    )

    # --- STAGE 1: Fast, Single-Criterion SPFA ---
    # Find the single cheapest path to every node.
    best_labels = {}
    queue = deque([start_label])
    
    spfa_start_time = time.time()
    print(f"Finding cheapest paths for source {source_node}...")

    while queue:
        u_label = queue.popleft()
        u_node, u_time = u_label[NODE_IDX], u_label[CURRENT_TIME_IDX]

        for v_node, arc_data in graph.adj[u_node].items():
            v_node_data = graph.nodes[v_node]
            v_type = v_node_data['type']
            new_label = None

            if v_type == 'T':
                dh_time = arc_data["time"]
                
                if u_label[NODE_IDX].startswith("c"):
                    if v_node_data["start_time"] < u_label[CURRENT_TIME_IDX] + timedelta(minutes=dh_time):
                        continue

                clock_at_u = v_node_data["start_time"] - timedelta(minutes=dh_time) if u_time is None else u_time
                if clock_at_u + timedelta(minutes=dh_time) > v_node_data["start_time"]: continue
                wait_time = (v_node_data["start_time"] - (clock_at_u + timedelta(minutes=dh_time))).total_seconds() / 60.0
                travel_time = v_node_data['planned_travel_time']
                new_total_time = u_label[TOTAL_TIME_IDX] + dh_time + wait_time + travel_time
                if new_total_time >= T_d: continue
                dh_dist = arc_data["dist"]
                travel_dist = dh_df.loc[v_node_data['start_cp'], v_node_data['end_cp']]
                new_dist_since_charge = u_label[DIST_SINCE_CHARGE_IDX] + dh_dist + travel_dist
                if new_dist_since_charge >= D:
                    
                    depot_cols = dh_df.filter(regex=r"^d\d+_\d+$").columns
                    best = dh_df.loc[graph.nodes[u_node]['end_cp'], depot_cols].idxmin()
                    cs = "c" + best[1:]
                    dist_cs = graph[u_node][cs]["dist"]
                    time_cs = graph[u_node][cs]["time"]
                    charge_minutes = (15.6 * ((u_label[DIST_SINCE_CHARGE_IDX] + dist_cs) / 20) / 30) * 60
                    new_clock = clock_at_u + timedelta(minutes=time_cs + charge_minutes)

                    if u_label[TOTAL_TIME_IDX] + time_cs + charge_minutes >= T_d: continue

                    weight_cs = graph[u_node][cs]["reduced_cost"]
                    new_label = (
                        cs,
                        u_label[TOTAL_WEIGHT_IDX] + weight_cs,
                        u_label[TOTAL_DIST_IDX] + dist_cs,
                        u_label[TOTAL_TIME_IDX] + time_cs + charge_minutes,
                        0.0,
                        0.0,
                        new_clock,
                        u_label[PATH_IDX] + (cs,),
                        u_label[TOTAL_DH_DIST_IDX] + dist_cs,
                        u_label[TOTAL_DH_TIME_IDX] + time_cs,
                        u_label[TOTAL_TRAVEL_DIST_IDX],
                        u_label[TOTAL_TRAVEL_TIME_IDX],
                        u_label[TOTAL_WAIT_TIME_IDX] + charge_minutes,
                        u_label[TOTAL_CHARGING_TIME_IDX] + charge_minutes
                    )

                    prev_label = best_labels.get(cs)

                    if prev_label is None or new_label[TOTAL_WEIGHT_IDX] < prev_label[TOTAL_WEIGHT_IDX]:
                        best_labels[cs] = new_label
                        queue.append(new_label)
                    continue                

                new_label = (
                    v_node,
                    u_label[TOTAL_WEIGHT_IDX] + arc_data['reduced_cost'],
                    u_label[TOTAL_DIST_IDX] + dh_dist + travel_dist,
                    new_total_time,
                    new_dist_since_charge,
                    u_label[TIME_SINCE_CHARGE_IDX] + dh_time + travel_time,
                    clock_at_u + timedelta(minutes=(dh_time + wait_time + travel_time)) ,
                    u_label[PATH_IDX] + (v_node,),
                    u_label[TOTAL_DH_DIST_IDX] + dh_dist,
                    u_label[TOTAL_DH_TIME_IDX] + dh_time,
                    u_label[TOTAL_TRAVEL_DIST_IDX] + travel_dist,
                    u_label[TOTAL_TRAVEL_TIME_IDX] + travel_time,
                    u_label[TOTAL_WAIT_TIME_IDX] + wait_time,
                    u_label[TOTAL_CHARGING_TIME_IDX]
                )

                prev_label = best_labels.get(v_node)
                if prev_label is None or new_label[TOTAL_WEIGHT_IDX] < prev_label[TOTAL_WEIGHT_IDX]:
                    best_labels[v_node] = new_label
                    if new_label not in queue:
                        queue.append(new_label)

    print(f"SPFA search for {source_node} finished in {time.time() - spfa_start_time:.2f}s.")

    # --- STAGE 2: POST-PROCESSING USING "GROUP BY FIRST TRIP" LOGIC ---
    print("Applying 'Group by First Trip' column selection logic...")
    grouped_by_first_trip = defaultdict(list)

    # print("*"*300)
    # print("BEST LABELS")
    # print("----------------")
    # print(best_labels)
    
    T = [n for n, d in graph.nodes(data=True) if d.get('type') == 'T']
    trip_routes = {}
    _i = 1

    for trip_id in T:
        trip_routes[trip_id] = list(filter(lambda x: x[PATH_IDX][1] == trip_id, best_labels.values()))

        best_cols = []

        for v in trip_routes[trip_id]:
            if v[PATH_IDX][-1].startswith("c"):
                continue
            full_path = v[PATH_IDX] + (source_node,)
            total_travel_dist = v[TOTAL_TRAVEL_DIST_IDX]
            total_travel_time = v[TOTAL_TRAVEL_TIME_IDX]
            total_wait_time = v[TOTAL_WAIT_TIME_IDX]
            total_charging_time = v[TOTAL_CHARGING_TIME_IDX]
            total_dh_dist = v[TOTAL_DH_DIST_IDX] + graph[v[PATH_IDX][-1]][source_node]["dist"]
            total_dh_time = v[TOTAL_DH_TIME_IDX] + graph[v[PATH_IDX][-1]][source_node]["time"]
            final_time = v[CURRENT_TIME_IDX] + timedelta(minutes=graph[v[PATH_IDX][-1]][source_node]["time"]) if v[CURRENT_TIME_IDX] else None
            recharging_freq = sum(1 for node in full_path if node.startswith("c"))
            cost = calculate_cost(
                recharging_freq=recharging_freq, travel_time=v[TOTAL_TRAVEL_TIME_IDX],
                dh_time=total_dh_time, waiting_time=v[TOTAL_WAIT_TIME_IDX]
            )
            trips_in_path = [node for node in full_path if graph.nodes[node].get('type') == 'T']
            alpha_sum = sum(duals['alpha'][graph.nodes[trip]['id']] for trip in trips_in_path)
            beta_val = duals['beta'][source_node]
            reduced_cost_manual = cost - alpha_sum + beta_val
            route = {
                "Path": list(full_path),
                "TravelDistance": total_travel_dist,
                "ReducedCost": reduced_cost_manual,
                "Cost": cost,
                "Data": {
                    "total_travel_time": total_travel_time,
                    "total_dh_dist": total_dh_dist,
                    "total_dh_time": total_dh_time,
                    "total_distance": total_travel_dist + total_dh_dist,
                    "total_wait_time": total_wait_time,
                    "total_charging_time": total_charging_time,
                    "final_time": final_time
                    }
                }
            best_cols.append(route)

        trip_routes[trip_id] = best_cols

    return trip_routes