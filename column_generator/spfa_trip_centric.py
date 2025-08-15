from collections import deque
from datetime import timedelta
import time
from initializer.inputs import problem_params

# --- Index Constants for Label Tuple ---
NODE_IDX = 0
TOTAL_WEIGHT_IDX = 1
TOTAL_DIST_IDX = 2
TOTAL_TIME_IDX = 3
DIST_SINCE_CHARGE_IDX = 4
TIME_SINCE_CHARGE_IDX = 5
CURRENT_TIME_IDX = 6
PATH_IDX = 7
TOTAL_DH_DIST_IDX = 8
TOTAL_DH_TIME_IDX = 9
TOTAL_TRAVEL_DIST_IDX = 10
TOTAL_TRAVEL_TIME_IDX = 11
TOTAL_WAIT_TIME_IDX = 12
TOTAL_CHARGING_TIME_IDX = 13


def run_spfa_for_paths(graph, source_node, D, T_d, dh_df):
    """
    This SPFA variant runs a depot-centric search and returns the entire
    collection of non-dominated labels (partial paths) found. It uses a
    relaxed, weight-only dominance check internally to generate a rich
    set of candidate paths.
    """
    start_label = (
        source_node, 0.0, 0.0, 0.0, 0.0, 0.0, None, (source_node,),
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    )

    best_labels = {node: [] for node in graph.nodes()}
    best_labels[source_node].append(start_label)
    queue = deque([start_label])
    
    consumption_rate = problem_params['consumption_rate_kWh_per_hour']
    charging_rate = problem_params['charging_rate_kW_per_hour']

    spfa_start_time = time.time()
    print(f"Extracting all paths for source {source_node}...")

    while queue:
        u_label = queue.popleft()
        u_node, u_time = u_label[NODE_IDX], u_label[CURRENT_TIME_IDX]

        for v_node, arc_data in graph.adj[u_node].items():
            v_node_data = graph.nodes[v_node]
            v_type = v_node_data['type']

            if v_type == 'T':
                dh_time = arc_data["time"]
                clock_at_u = v_node_data["start_time"] - timedelta(minutes=dh_time) if u_time is None else u_time
                if clock_at_u + timedelta(minutes=dh_time) > v_node_data["start_time"]: continue
                wait_time = (v_node_data["start_time"] - (clock_at_u + timedelta(minutes=dh_time))).total_seconds() / 60.0
                travel_time = v_node_data['planned_travel_time']
                new_total_time = u_label[TOTAL_TIME_IDX] + dh_time + wait_time + travel_time
                if new_total_time > T_d: continue
                dh_dist = arc_data["dist"]
                travel_dist = dh_df.loc[v_node_data['start_cp'], v_node_data['end_cp']]
                new_dist_since_charge = u_label[DIST_SINCE_CHARGE_IDX] + dh_dist + travel_dist
                if new_dist_since_charge > D: continue

                new_label = (
                    v_node, u_label[TOTAL_WEIGHT_IDX] + arc_data['reduced_cost'],
                    u_label[TOTAL_DIST_IDX] + dh_dist + travel_dist, new_total_time, new_dist_since_charge,
                    u_label[TIME_SINCE_CHARGE_IDX] + dh_time + wait_time + travel_time,
                    v_node_data['end_time'], u_label[PATH_IDX] + (v_node,),
                    u_label[TOTAL_DH_DIST_IDX] + dh_dist, u_label[TOTAL_DH_TIME_IDX] + dh_time,
                    u_label[TOTAL_TRAVEL_DIST_IDX] + travel_dist, u_label[TOTAL_TRAVEL_TIME_IDX] + travel_time,
                    u_label[TOTAL_WAIT_TIME_IDX] + wait_time, u_label[TOTAL_CHARGING_TIME_IDX]
                )
                
                if not any(e[TOTAL_WEIGHT_IDX] <= new_label[TOTAL_WEIGHT_IDX] for e in best_labels[v_node]):
                    best_labels[v_node] = [e for e in best_labels[v_node] if not (new_label[TOTAL_WEIGHT_IDX] <= e[TOTAL_WEIGHT_IDX])] + [new_label]
                    queue.append(new_label)

            elif v_type == 'C':
                time_to_cs = arc_data['time']
                consumption_kWh = (consumption_rate / 60) * (u_label[TIME_SINCE_CHARGE_IDX] + time_to_cs)
                charge_minutes = (consumption_kWh / charging_rate) * 60
                new_total_time = u_label[TOTAL_TIME_IDX] + time_to_cs + charge_minutes
                if new_total_time > T_d: continue
                dist_to_cs = arc_data['dist']
                if u_label[DIST_SINCE_CHARGE_IDX] + dist_to_cs > D: continue

                new_label = (
                    v_node, u_label[TOTAL_WEIGHT_IDX] + arc_data['reduced_cost'],
                    u_label[TOTAL_DIST_IDX] + dist_to_cs, new_total_time,
                    0.0, 0.0, u_time + timedelta(minutes=(time_to_cs + charge_minutes)),
                    u_label[PATH_IDX] + (v_node,),
                    u_label[TOTAL_DH_DIST_IDX] + dist_to_cs, u_label[TOTAL_DH_TIME_IDX] + time_to_cs,
                    u_label[TOTAL_TRAVEL_DIST_IDX], u_label[TOTAL_TRAVEL_TIME_IDX],
                    u_label[TOTAL_WAIT_TIME_IDX] + charge_minutes, u_label[TOTAL_CHARGING_TIME_IDX] + charge_minutes
                )

                if not any(e[TOTAL_WEIGHT_IDX] <= new_label[TOTAL_WEIGHT_IDX] for e in best_labels[v_node]):
                    best_labels[v_node] = [e for e in best_labels[v_node] if not (new_label[TOTAL_WEIGHT_IDX] <= e[TOTAL_WEIGHT_IDX])] + [new_label]
                    queue.append(new_label)

    spfa_end_time = time.time()
    print(f"Path extraction for {source_node} finished in {spfa_end_time - spfa_start_time:.2f}s.")
    
    return best_labels
