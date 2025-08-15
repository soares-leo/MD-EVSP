import networkx as nx
import pandas as pd
from datetime import timedelta
from initializer.inputs import problem_params, depots, lines_info, cp_depot_distances
from initializer.utils import summarize_cp_locations, calculate_cp_distances, transform_cp_depot_distances, merge_distances_dicts, make_deadhead_df, make_deadhead_times_df


class GraphBuilder:
    """Class to build a directed graph of trips, depots, and charging stations with deadhead arcs."""

    def __init__(self, timetable_path: str, used_depots: list):
        self.timetable_path = timetable_path
        self.used_depots = used_depots
        # Initialize core data and graph
        self.timetables = self.load_timetables()
        self.dh_df, self.dh_times_df = self.build_deadhead_data()
        self.G = self.initialize_graph()
        # Placeholders for nodes
        self.trip_nodes = {}
        self.depot_ids = []
        self.cs_ids = []
        # Build full graph
        self.build_graph()

    def load_timetables(self) -> pd.DataFrame:
        """Load timetables CSV with parsed departure times."""
        return pd.read_csv(self.timetable_path, parse_dates=["departure_time"])

    def build_deadhead_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate deadhead distance and time DataFrames."""
        summary = summarize_cp_locations(lines_info)
        cp_dist = calculate_cp_distances(summary, lines_info)
        transformed = transform_cp_depot_distances(cp_depot_distances)
        dh_dict = merge_distances_dicts(transformed, cp_dist)
        dh_df = make_deadhead_df(dh_dict)
        dh_times = make_deadhead_times_df(20, dh_df)
        return dh_df, dh_times

    def initialize_graph(self) -> nx.DiGraph:
        """Initialize an empty directed NetworkX graph."""
        return nx.DiGraph()

    def add_trip_nodes(self):
        """Add trip nodes and metadata to the graph."""
        for _, row in self.timetables.iterrows():
            tid = row['trip_id']
            attrs = {
                'id': tid,
                'start_cp': row['start_cp_id'],
                'end_cp': row['dest_cp_id'],
                'start_time': row['departure_time'],
                'planned_travel_time': row['planned_travel_time'],
                'end_time': row['departure_time'] + timedelta(minutes=row['planned_travel_time']),
                'type': 'T'
            }
            self.trip_nodes[tid] = attrs
            self.G.add_node(tid, **attrs)

    def add_depot_and_cs_nodes(self):
        """Add depot (K) and charging station (C) nodes."""
        filtered_depots = {key: value for key, value in depots.items() if key in self.used_depots}
        self.depot_ids = list(filtered_depots.keys())
        self.cs_ids = [d.replace('d', 'c') for d in self.depot_ids]
        for d in self.depot_ids:
            self.G.add_node(d, type='K')
        for c in self.cs_ids:
            self.G.add_node(c, type='C')

    def add_arcs_out(self):
        """Add arcs between depots and trips (A_out and A_in)."""
        for d in self.depot_ids:
            for tid, data in self.trip_nodes.items():
                dist_from_d = float(self.dh_df.loc[d, data['start_cp']])
                time_from_d = float(self.dh_times_df.loc[d, data['start_cp']])
                dist_to_d = float(self.dh_df.loc[data['end_cp'], d])
                time_to_d = float(self.dh_times_df.loc[data['end_cp'], d])
                self.G.add_edge(d, tid, type='A_out', dist=dist_from_d, time=time_from_d)
                self.G.add_edge(tid, d, type='A_in', dist=dist_to_d, time=time_to_d)
    
    """
    def add_dh_trip_arcs(self):
        items = list(self.trip_nodes.items())
        for i, (i_id, i_data) in enumerate(items):
            for j, (j_id, j_data) in enumerate(items):
                if i == j:
                    continue
                dist = float(self.dh_df.loc[i_data['end_cp'], j_data['start_cp']])
                tm = float(self.dh_times_df.loc[i_data['end_cp'], j_data['start_cp']])
                if i_data['end_time'] + timedelta(minutes=tm) <= j_data['start_time']:
                    self.G.add_edge(i_id, j_id, type='A_dh_trips', dist=dist, time=tm)
    """

    def add_dh_trip_arcs(self):
        """Add deadhead arcs between sequential trips, pruning excessively long waits."""
        # This is a tunable parameter. A lower value creates a sparser graph,
        # which speeds up the subproblem, but risks pruning optimal connections.
        # 180 minutes (3 hours) is a reasonable starting point.
        MAX_WAIT_TIME_MINS = 960.0

        items = list(self.trip_nodes.items())
        for i, (i_id, i_data) in enumerate(items):
            for j, (j_id, j_data) in enumerate(items):
                if i == j:
                    continue
                
                # Check for temporal precedence first
                if i_data['end_time'] > j_data['start_time']:
                    continue

                dist = float(self.dh_df.loc[i_data['end_cp'], j_data['start_cp']])
                tm = float(self.dh_times_df.loc[i_data['end_cp'], j_data['start_cp']])
                
                # Check if the connection is feasible in time
                if i_data['end_time'] + timedelta(minutes=tm) <= j_data['start_time']:
                    # NEW: Check if the wait time is excessive
                    # wait_time = (j_data['start_time'] - (i_data['end_time'] + timedelta(minutes=tm))).total_seconds() / 60.0
                    
                    # if wait_time <= MAX_WAIT_TIME_MINS:
                    self.G.add_edge(i_id, j_id, type='A_dh_trips', dist=dist, time=tm)


    def add_cs_arcs(self):
        """Add charging station arcs to and from trips."""
        for c in self.cs_ids:
            for tid, data in self.trip_nodes.items():
                dist_from_c = float(self.dh_df.loc[c.replace("c", "d"), data['start_cp']])
                time_from_c = float(self.dh_times_df.loc[c.replace("c", "d"), data['start_cp']])
                dist_to_c = float(self.dh_df.loc[data['end_cp'], c.replace("c", "d")])
                time_to_c = float(self.dh_times_df.loc[data['end_cp'], c.replace("c", "d")])
                self.G.add_edge(c, tid, type='A_cs_from', dist=dist_from_c, time=time_from_c)
                self.G.add_edge(tid, c, type='A_cs_to', dist=dist_to_c, time=time_to_c)

    def build_graph(self):
        """Execute all steps to populate the graph."""
        self.add_trip_nodes()
        self.add_depot_and_cs_nodes()
        self.add_arcs_out()
        self.add_dh_trip_arcs()
        self.add_cs_arcs()
        return self.G


# if __name__ == '__main__':
#     builder = GraphBuilder('initializer/files/timetables.csv')
#     print(builder.G)


# import networkx as nx
# import pandas as pd
# from datetime import timedelta
# from inputs import *
# from utils import *

# # === Load trips ===
# timetables = pd.read_csv("initializer/files/timetables.csv", parse_dates=["departure_time"])

# # === Parameters ===
# speed_kmph = 20

# Ct = (problem_params['battery_capacity_kWh'] * (1 - problem_params['soc_lower_bound'])) / \
#      problem_params['charging_rate_kW_per_hour'] * 60  # charging time in minutes

# depots_ids = list(depots.keys())
# charging_station_ids = list(map(lambda x: x.replace("d", "c"), depots_ids))

# cp_locations_summary = summarize_cp_locations(lines_info)
# cp_distances = calculate_cp_distances(cp_locations_summary, lines_info)
# transformed_cp_depot_distances = transform_cp_depot_distances(cp_depot_distances)
# dh_dict = merge_distances_dicts(transformed_cp_depot_distances, cp_distances)
# dh_df = make_deadhead_df(dh_dict)
# dh_times_df = make_deadhead_times_df(20, dh_df)

# print(dh_times_df)

# # === Initialize directed graph ===
# G = nx.DiGraph()

# # === Add trip nodes ===
# trip_nodes = {}
# for idx, row in timetables.iterrows():
#     trip_id = row["trip_id"]
#     start_cp = row["start_cp_id"]
#     end_cp = row["dest_cp_id"]
#     start_time = row["departure_time"]
#     end_time = start_time + timedelta(minutes=row["planned_travel_time"])

#     trip_nodes[trip_id] = {
#         "id": trip_id,
#         "start_cp": start_cp,
#         "end_cp": end_cp,
#         "start_time": start_time,
#         "end_time": end_time,

#     }
#     G.add_node(trip_id, **trip_nodes[trip_id], type="T")

# # === Add depot and charging station nodes ===
# for depot in depots_ids:
#     G.add_node(depot, type="K")

# for cs in charging_station_ids:
#     G.add_node(cs, type="C")

# # # === Add arcs ===
# # A_out: depot → trip
# for depot in depots_ids:
#     for trip_id, trip_data in trip_nodes.items():
#         ss = timetables[timetables.trip_id == trip_id]["start_cp_id"]
#         dist = float(dh_df.loc[depot,ss])
#         time = float(dh_times_df.loc[depot,ss])
#         G.add_edge(depot, trip_id, type="A_out", dist=dist, time=time)
#         G.add_edge(trip_id, depot, type="A_in", dist=dist, time=time)

# # A_dh: trip → trip
# trip_list = list(trip_nodes.items())
# for i, (trip_i_id, ti) in enumerate(trip_list):
#     for j, (trip_j_id, tj) in enumerate(trip_list):
#         if i == j:
#             continue
#         dist = float(dh_df.loc[ti["end_cp"], tj["start_cp"]])
#         time = float(dh_times_df.loc[ti["end_cp"], tj["start_cp"]])
#         if (ti["end_time"] + timedelta(minutes=time)) <= tj["start_time"]:
            
#             G.add_edge(trip_i_id, trip_j_id, type="A_dh_trips", dist=dist, time=time)

# # A_dh: trip → cs, cs → trip
# for trip_id, t in trip_nodes.items():
#     for cs in charging_station_ids:
#         G.add_edge(trip_id, cs, type="A_cs_to")
#         G.add_edge(cs, trip_id, type="A_cs_from")

# print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# # import matplotlib.pyplot as plt
# # # === Visualize the graph (simplified layout) ===
# # plt.figure(figsize=(12, 8))
# # pos = nx.spring_layout(G, seed=42, k=0.15)
# # nx.draw(G, pos, node_size=20, arrows=True, with_labels=False)
# # plt.title("Connection Network Graph")
# # plt.tight_layout()
# # plt.show()

# print(G)

# # === Reduced cost function ===
# def compute_reduced_cost(i, j, G, alpha, beta, fixed_cost=0):
#     arc_data = G.get_edge_data(i, j)
#     if arc_data is None:
#         return float('inf')

#     base_cost = arc_data.get('travel_time', 0) + fixed_cost
#     reduced = base_cost

#     # subtract dual values only when nodes represent trips or depots
#     if G.nodes[j].get('type') is None:  # j is a trip
#         reduced -= alpha.get(j, 0)
#     if G.nodes[i].get('type') == 'depot':
#         reduced -= beta.get(i, 0)

#     return reduced
