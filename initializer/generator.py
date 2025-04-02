from inputs import *
from utils import *
from datetime import datetime, timedelta

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
        self.timetables.to_csv("timetables.csv")
        self.timetables['covered'] = False
        self.ti = None
        self.dist = 0
        self.time = 0
        self.time_since_recharge = 0
        self.max_time = 360
        self.max_dist = 60
        self.recharging_rate = 30
        self.consumption_rate = 15.6
        self.route = []
        self.initial_solution = {}

    def generate_initial_set(self):
        i = 0
        # print()
        # print("############# GENERATOR STARTED #############")
        # print("---------------------------------------------")
        while not self.timetables[self.timetables.covered == False].empty:
            # if i == 10:
            #     break
            # print()
            # print("ROUTE:", i)
            # print("---------------")
            # print()    
            
            if (ti := self.initialize(i)) is None:
                # print()
                # print("Trip 0:")
                # print("\tNone")
                # print()
                break
            # print()
            # print("Trip 0:")
            # print(f"{ti}")
            # print()
            #if i < 1:
                # print()
                # print(self.route)
                # print(self.dist)
                # print(self.time)
                # print()                
            continue_loop = True
            _i = 1
            while continue_loop:
                # if _i == 10:
                #     break
                ti, continue_loop = self.update_route(ti)
                # print()
                # print(f"Trip {_i}:")
                # print(f"{ti}")
                # print()
                _i += 1
            
            self.initial_solution[f"Route_{i}"] = (self.route, self.dist, self.time)
            # print("Route:", self.route)
            # print("Total_distance:", self.dist)
            # print("Total_time:", self.time)

            self.route = []
            self.time = 0
            self.dist = 0
            self.time_since_recharge = 0
            i+=1

        for k, v in self.initial_solution.items():
            print(f"{k}: {v}")

    def initialize(self, i):

        if (ti := get_earliest_trip(self.timetables)) is None:
            self.route.append(self.route[0])
            return None

        departure_depot = get_nearest_depot(ti["start_cp_id"], self.depots, self.cp_depot_distances)
        
        if departure_depot is None:
            self.route.append(self.route[0])
            return None
        
        self.depots[departure_depot]['departed'] += 1
        self.route.extend([departure_depot, ti['trip_id']])

        self.timetables.loc[self.timetables.trip_id == ti.trip_id, 'covered'] = True

        # if i < 1:
        #     print()
        #     print(self.dist)
        #     print(self.time)
        #     print()

        self.dist += self.dh_df.loc[ti.start_cp_id, departure_depot] + lines_info[ti["line"]]["cp_distance_km"]
        self.time += self.dh_times_df.loc[ti.start_cp_id, departure_depot] + ti['planned_travel_time']
        self.time_since_recharge += self.time

        # if i < 1:
        #     print()
        #     print("From depot to cp:", self.dh_df.loc[ti.start_cp_id, departure_depot])
        #     print("trip_distance:",lines_info[ti["line"]]["cp_distance_km"])

        return ti

    def update_route(self, ti=None):
        if ti is None:
            ti = self.timetables[self.timetables.trip_id == self.route[-1]].iloc[0]
            tj = select_next_trip(self.timetables, self.dh_times_df, ti.trip_id)
        else:
            tj = select_next_trip(self.timetables, self.dh_times_df, None, ti)

        if tj is None:
            self.route.append(self.route[0])
            return None, False
        
        tij_dist = self.dh_df.loc[ti.dest_cp_id, tj.start_cp_id]
        tij_time = self.dh_times_df.loc[ti.dest_cp_id, tj.start_cp_id]
        tj_dist = self.dh_df.loc[tj.start_cp_id, tj.dest_cp_id]
        pred_dist = self.dist + tij_dist + tj_dist
        pred_time = self.time + tij_time + tj.planned_travel_time
        
        if pred_dist <= self.max_dist and pred_time <= self.max_time:
            self.route.append(tj.trip_id)
            self.timetables.loc[self.timetables.trip_id == tj.trip_id, 'covered'] = True
            self.time = pred_time
            self.dist = pred_dist
            self.time_since_recharge += tij_time + tj.planned_travel_time
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
            
            self.time_since_recharge = 0

            self.time += charging_time

            if self.time >= self.max_time:
                self.route.append(self.route[0])
                # Aqui precisa fazer mais uma pred time e ESPECIALMENTE pred dist (como condiçao AND do if, ou até mesmo como sendo a unica condicao do if) pra saber se no lugar da estacao nao precisa de um depósito.
                return None, False
            else:

                self.dist = 0
                self.route.append(charging_station)
         
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

                ## NÃO DELETAR A LINHA ABAIXO!!!
                ## OBS: step 2.2 inconsistente, por isso foi modificado. Não tem re-checking de Time. Procura em todas as timetables, quando poderia procurar primeiro no cp onde está.

                #print(ti)
                return ti, True
        else:
            self.route.append(self.route[0])
            return None, False

generator = Generator(lines_info=lines_info, cp_depot_distances=cp_depot_distances, depots=depots)
generator.generate_initial_set()