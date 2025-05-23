from .inputs import *
from .utils import *
from datetime import timedelta
import numpy as np
import pandas as pd

# Initial parameters

class Generator:
    def __init__(self, lines_info, cp_depot_distances, depots):
        self.lines_info = lines_info
        self.cp_depot_distances = cp_depot_distances
        self.cp_locations_summary = summarize_cp_locations(lines_info)
        self.cp_distances = calculate_cp_distances(self.cp_locations_summary, lines_info)
        self.transformed_cp_depot_distances = transform_cp_depot_distances(cp_depot_distances)
        self.dh_dict = merge_distances_dicts(self.transformed_cp_depot_distances, self.cp_distances)
        self.dh_df = make_deadhead_df(self.dh_dict)
        self.dh_times_df = make_deadhead_times_df(20, self.dh_df)
        self.depots = depots
        tt_l4 = generate_timetable("4", 290)
        tt_l59 = generate_timetable("59", 100)
        tt_l60 = generate_timetable("60", 120)
        self.timetables = pd.concat([tt_l4, tt_l59, tt_l60], ignore_index=True)
        self.timetables.to_csv("initializer/files/timetables.csv")
        self.timetables['covered'] = False
        self.ti = None
        self.dist = 0
        self.time = 0
        self.route_total_dist = 0
        self.route_total_time = 0
        self.time_since_recharge = 0
        self.max_time = 19*60
        self.max_dist = 120
        self.recharging_rate = 30
        self.consumption_rate = 15.6
        self.route = []
        self.initial_solution = {}
        self.total_dh_dist = 0
        self.total_dh_time = 0
        self.total_travel_dist = 0
        self.total_travel_time = 0
        self.total_wait_time = 0
        self.total_charging_time = 0
        self.used_depots = []

    def generate_initial_set(self):
        i = 0
        while not self.timetables[self.timetables.covered == False].empty:
            if len(str(i)) == 1:
                comp = "00"
            elif len(str(i)) == 2:
                comp = "0"
            else:
                comp = ""
            print()      
            print(f"+-----------+-----------+-----------+ ROUTE {comp + str(i)} +-----------+-----------+-----------+")
            print()
            if (ti := self.initialize(i)) is None:
                break              
            continue_loop = True
            _i = 1
            while continue_loop:
                
                print("ITERATION", _i-1)
                try:
                    print("Trip id:", ti.trip_id)
                    print("Departure time:", ti.departure_time)
                    print("Planned travel time:", ti.planned_travel_time)
                except:
                    print("Recharge in cp of:", self.route[-1])
                print("Dist:", self.dist)
                print("Time:", self.time)
                print("Time since racharge:", self.time_since_recharge)
                print("Deadhead distance:", self.total_dh_dist)
                print("Deadhead time:", self.total_dh_time)
                print("Total travel distance:", self.total_travel_dist)
                print("Total travel time:", self.total_travel_time)
                print("Total wait time:", self.total_wait_time)
                print("Current time:", self.current_time)
                print("Route:")
                print(f"\t{self.route}")
                print()
                
                ti, continue_loop = self.update_route(ti)
                _i += 1

            recharging_freq = len(list(filter(lambda x: x[0] == "c", self.route)))

            route_cost = calculate_cost(
                recharging_freq, self.total_travel_time, self.total_dh_time, self.total_wait_time)
            
            self.initial_solution[f"Route_{i}"] = {
                "Path": self.route,
                "Cost": route_cost,
                "Data": {
                    "total_dh_dist": self.total_dh_dist,
                    "total_dh_time": self.total_dh_time,
                    "total_travel_dist": self.total_travel_dist,
                    "total_travel_time": self.total_travel_time,
                    "total_wait_time": self.total_wait_time,
                    "final_time": self.current_time
                }
            }

            print("Last node:", self.route[-1])
            print("Dist:", self.dist)
            print("Time:", self.time)
            print("Time since racharge:", self.time_since_recharge)
            print("Deadhead distance:", self.total_dh_dist)
            print("Deadhead time:", self.total_dh_time)
            print("Total travel distance:", self.total_travel_dist)
            print("Total travel time:", self.total_travel_time)
            print("Total wait time:", self.total_wait_time)
            print("Current time:", self.current_time)
            print("Route:")
            print(f"\t{self.route}")
            print()

            self.route = []
            self.time = 0
            self.dist = 0
            self.time_since_recharge = 0
            self.total_dh_dist = 0
            self.total_dh_time = 0
            self.total_travel_dist = 0
            self.total_travel_time = 0
            self.total_wait_time = 0
            self.total_charging_time = 0
            
            i += 1

        for k, v in self.initial_solution.items():
            print(f"\n{k}:")
            print("Path:", v["Path"])
            print("Cost:", v["Cost"])
            print("Data:")
            for _k, _v in self.initial_solution[k]["Data"].items():
                print(f"\t{_k}: {_v}")
        
        self.timetables.to_csv("initializer/files/timetables_covered.csv", index=False)
        pd.DataFrame(self.depots).to_csv("initializer/files/depots_final_picture.csv")
        return self.initial_solution, self.used_depots

    def refresh_data(self):
        nodes = self.route[-2:]
        if nodes[0][0] == "d" and nodes[1][0] == "l":
            ti = self.timetables[self.timetables.trip_id == nodes[1]].iloc[0]
            dh_dist_ij = self.dh_df.loc[nodes[0], ti.start_cp_id]
            dh_time_ij = self.dh_times_df.loc[nodes[0], ti.start_cp_id]
            travel_dist = self.dh_df.loc[ti.start_cp_id, ti.dest_cp_id]
            travel_time = ti.planned_travel_time
            wait_time = 0
            print("Extra info - wait time:", wait_time)
            print()
            equalizer = 0
            charging_time = 0
            self.current_time = ti.departure_time + timedelta(minutes=travel_time) 
        if (nodes[0][0] == "l" or nodes[0][0] == "c") and nodes[1][0] == "l":
            if nodes[0][0] == "c":
                ti = self.timetables[self.timetables.trip_id == self.route[-3]].iloc[0]
            else:
                ti = self.timetables[self.timetables.trip_id == nodes[0]].iloc[0]
            tj = self.timetables[self.timetables.trip_id == nodes[1]].iloc[0]
            dh_dist_ij = self.dh_df.loc[ti.dest_cp_id, tj.start_cp_id]
            dh_time_ij = self.dh_times_df.loc[ti.dest_cp_id, tj.start_cp_id]
            travel_dist = self.dh_df.loc[tj.start_cp_id, tj.dest_cp_id]
            travel_time = tj.planned_travel_time
            arrival_time_at_tj = self.current_time + timedelta(minutes=dh_time_ij)
            print("Extra info - arrival time at tj:", arrival_time_at_tj)
            time_diff = tj.departure_time - arrival_time_at_tj
            wait_time = time_diff.total_seconds() / 60
            print("Extra info - wait time:", wait_time)
            print()
            equalizer = 0
            charging_time = 0

            self.current_time = tj.departure_time + timedelta(minutes=travel_time) 

        if nodes[0][0] == "l" and nodes[1][0] == "c":
            dh_dist_ij = 0
            dh_time_ij = 0
            travel_dist = 0
            travel_time = 0
            
            charging_time = self.consumption_rate * (self.time_since_recharge / self.recharging_rate)
            wait_time = charging_time

            print("Extra info - wait time:", wait_time)
            print()

            equalizer = self.time_since_recharge

            self.current_time += timedelta(minutes = dh_time_ij + travel_time + wait_time)

        if nodes[0][0] == "l" and nodes[1][0] == "d":
            tj = self.timetables[self.timetables.trip_id == nodes[0]].iloc[0]
            dh_dist_ij = self.dh_df.loc[tj.dest_cp_id, nodes[1]]
            dh_time_ij = self.dh_times_df.loc[tj.dest_cp_id, nodes[1]]
            travel_dist = 0
            travel_time = 0
            wait_time = 0
            print("Extra info - wait time:", wait_time)
            print()
            equalizer = 0
            charging_time = 0
            
            self.current_time +=  timedelta(minutes=dh_time_ij)

        self.total_dh_dist += dh_dist_ij
        self.total_dh_time += dh_time_ij
        self.total_travel_dist += travel_dist
        self.total_travel_time += travel_time
        self.dist += dh_dist_ij + travel_dist
        self.time += dh_time_ij + travel_time + wait_time
        self.time_since_recharge += dh_time_ij + travel_time - equalizer
        self.total_wait_time += wait_time
        self.total_charging_time += charging_time
    def initialize(self, i):

        if (ti := get_earliest_trip(self.timetables)) is None:
            self.route.append(self.route[0])
            self.refresh_data()
            return None

        departure_depot = get_nearest_depot(ti["start_cp_id"], self.depots, self.cp_depot_distances)
        
        if departure_depot is None:
            self.route.append(self.route[0])
            self.refresh_data()
            return None
        
        self.depots[departure_depot]['departed'] += 1
        self.route.extend([departure_depot, ti['trip_id']])
        self.used_depots.append(departure_depot)

        self.refresh_data()

        self.timetables.loc[self.timetables.trip_id == ti.trip_id, 'covered'] = True

        return ti

    def update_route(self, ti=None):
        if ti is None:
            ti = self.timetables[self.timetables.trip_id == self.route[-1]].iloc[0]
            tj = select_next_trip(self.current_time, self.timetables, self.dh_times_df, ti.trip_id)
        else:
            tj = select_next_trip(self.current_time, self.timetables, self.dh_times_df, None, ti)

        if tj is None:
            self.route.append(self.route[0])
            self.refresh_data()
            return None, False
        
        tij_dist = self.dh_df.loc[ti.dest_cp_id, tj.start_cp_id]
        tij_time = self.dh_times_df.loc[ti.dest_cp_id, tj.start_cp_id]
        tj_dist = self.dh_df.loc[tj.start_cp_id, tj.dest_cp_id]

        pred_dist = self.dist + tij_dist + tj_dist
        time_diff = tj.departure_time - (self.current_time + timedelta(tij_time))
        time_diff = time_diff.total_seconds() / 60 
        pred_time = self.time + tij_time + time_diff + tj.planned_travel_time
        
        if pred_dist <= self.max_dist and pred_time <= self.max_time:
            self.route.append(tj.trip_id)
            self.refresh_data()
            self.timetables.loc[self.timetables.trip_id == tj.trip_id, 'covered'] = True

            return tj, True
        elif pred_dist > self.max_dist and pred_time <= self.max_time:
            filtered_cols = [
                col for col in self.dh_df.columns
                if col.lower().startswith("d")
            ]

            row_values = self.dh_df.loc[ti.dest_cp_id, filtered_cols]
            zero_col = row_values.idxmin() if (row_values == 0.0).any() else None
            zero_val = 0.0 if zero_col else None

            depot_aka_charging_station = (zero_col, zero_val)

            charging_station = depot_aka_charging_station[0].replace("d", "c")

            charging_time = self.consumption_rate * (self.time_since_recharge / self.recharging_rate)

            pred_current_time = self.current_time + timedelta(minutes=charging_time)

            ti = pd.Series(
                {
                    "line": ti.line,
                    "start_cp": ti.start_cp,
                    "start_cp_id": ti.start_cp_id,
                    "dest_cp": ti.dest_cp,
                    "dest_cp_id": ti.dest_cp_id,
                    "departure_time": ti.departure_time + timedelta(minutes=ti.planned_travel_time) + timedelta(minutes=charging_time), # horario da primeira trip + time + charging time
                    "planned_travel_time": 0,
                }
            )

            tj = select_next_trip(pred_current_time, self.timetables, self.dh_times_df, None, ti)

            if tj is None:
                self.route.append(self.route[0])
                self.refresh_data()
                return None, False
            
            tij_time = self.dh_times_df.loc[ti.dest_cp_id, tj.start_cp_id]
            pred_time = self.time + charging_time + tij_time + tj.planned_travel_time

            if pred_time >= self.max_time:
                #ESSE CHECKING NAO ACONTECE NO ALGORITMO OFICIAL!            
                self.route.append(self.route[0])
                self.refresh_data()
                return None, False
            
            else:

                self.route.append(charging_station)
                self.refresh_data()
                self.dist = 0
                self.route.append(tj.trip_id)
                self.refresh_data()
                self.timetables.loc[self.timetables.trip_id == tj.trip_id, 'covered'] = True
         
                ## NÃO DELETAR A LINHA ABAIXO!!!
                ## OBS: step 2.2 inconsistente, por isso foi modificado. Não tem re-checking de Time. Procura em todas as timetables, quando poderia procurar primeiro no cp onde está.

                #print(ti)
                return None, True
        else:
            self.route.append(self.route[0])
            self.refresh_data()
            return None, False


# UTILIDADES PARA TESTES DE CODIGO:
# generator = Generator(lines_info=lines_info, cp_depot_distances=cp_depot_distances, depots=depots)
# initial_solution = generator.generate_initial_set()

# concat_routes = []

# for k, v in initial_solution.items():
#     concat_routes.extend(v["Path"])

# cleaned_trips = list(filter(lambda x: x[0] not in ["c", "d"], concat_routes))

# unique_trips = np.unique(cleaned_trips)

# duplicates = {}
# helper = []

# cleaned_trips_copy = cleaned_trips.copy()

# for trip in cleaned_trips:
#     if trip not in helper:
#         helper.append(trip)
#     else:
#         if duplicates.get(trip):
#             duplicates[trip] += 1
#         else:
#             duplicates[trip] = 1

# print("Duplicated trips:", duplicates)