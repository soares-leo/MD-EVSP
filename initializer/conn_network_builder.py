import networkx as nx
import pandas as pd
from datetime import timedelta
from inputs import *
from utils import *

# === Load trips ===
timetables = pd.read_csv("initializer/files/timetables.csv", parse_dates=["departure_time"])

# === Parameters ===
speed_kmph = 20

Ct = (problem_params['battery_capacity_kWh'] * (1 - problem_params['soc_lower_bound'])) / \
     problem_params['charging_rate_kW_per_hour'] * 60  # charging time in minutes

depots_ids = list(depots.keys())
charging_station_ids = list(map(lambda x: x.replace("d", "c"), depots_ids))

cp_locations_summary = summarize_cp_locations(lines_info)
cp_distances = calculate_cp_distances(cp_locations_summary, lines_info)
transformed_cp_depot_distances = transform_cp_depot_distances(cp_depot_distances)
dh_dict = merge_distances_dicts(transformed_cp_depot_distances, cp_distances)
dh_df = make_deadhead_df(dh_dict)
dh_times_df = make_deadhead_times_df(20, dh_df)

# === Initialize directed graph ===
G = nx.Graph()

# === Add trip nodes ===
trip_nodes = {}
for idx, row in timetables.iterrows():
    trip_id = row["trip_id"]
    start_cp = row["start_cp_id"]
    end_cp = row["dest_cp_id"]
    start_time = row["departure_time"]
    end_time = start_time + timedelta(minutes=row["planned_travel_time"])

    trip_nodes[trip_id] = {
        "id": trip_id,
        "start_cp": start_cp,
        "end_cp": end_cp,
        "start_time": start_time,
        "end_time": end_time,

    }
    G.add_node(trip_id, **trip_nodes[trip_id], type="T")

# === Add depot and charging station nodes ===
for depot in depots_ids:
    G.add_node(depot, type="K")

for cs in charging_station_ids:
    G.add_node(cs, type="C")

# # === Add arcs ===
# A_out: depot → trip
for depot in depots_ids:
    for trip_id, trip_data in trip_nodes.items():
        G.add_edge(depot, trip_id, type="A_out")
        G.add_edge(trip_id, depot, type="A_in")

# A_dh: trip → trip
trip_list = list(trip_nodes.items())
for i, (trip_i_id, ti) in enumerate(trip_list):
    for j, (trip_j_id, tj) in enumerate(trip_list):
        if i == j:
            continue
        travel_time = dh_times_df.loc[ti["end_cp"], tj["start_cp"]]
        if (ti["end_time"] + timedelta(minutes=travel_time)) <= tj["start_time"]:
            G.add_edge(trip_i_id, trip_j_id, type="A_dh_trips")

# A_dh: trip → cs, cs → trip
for trip_id, t in trip_nodes.items():
    for cs in charging_station_ids:
        G.add_edge(trip_id, cs, type="A_cs_to")
        G.add_edge(cs, trip_id, type="A_cs_from")

print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# import matplotlib.pyplot as plt
# # === Visualize the graph (simplified layout) ===
# plt.figure(figsize=(12, 8))
# pos = nx.spring_layout(G, seed=42, k=0.15)
# nx.draw(G, pos, node_size=20, arrows=True, with_labels=False)
# plt.title("Connection Network Graph")
# plt.tight_layout()
# plt.show()

# === Reduced cost function ===
def compute_reduced_cost(i, j, G, alpha, beta, fixed_cost=0):
    arc_data = G.get_edge_data(i, j)
    if arc_data is None:
        return float('inf')

    base_cost = arc_data.get('travel_time', 0) + fixed_cost
    reduced = base_cost

    # subtract dual values only when nodes represent trips or depots
    if G.nodes[j].get('type') is None:  # j is a trip
        reduced -= alpha.get(j, 0)
    if G.nodes[i].get('type') == 'depot':
        reduced -= beta.get(i, 0)

    return reduced
