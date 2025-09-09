from .inputs import *
from .utils import *
from datetime import timedelta
import time
import pandas as pd

# Initial parameters

class InstanceGenerator:
    def __init__(self, lines_info, trips_configs=[]):
        self.lines_info = lines_info
        self.trips_configs = trips_configs
        
    def generate_instance(self):
        trips_str = ("_").join([f'l{str(dic["line"])}_{dic["total_trips"]}' for dic in self.trips_configs])
        self.instance_name = f"instance_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}_l4_{trips_str}"
        start_time = time.time()
        generated_timetables = [generate_timetable_v3(lines_info, **configs) for configs in self.trips_configs]
        end_time = time.time()
        self.timetables_generation_time = end_time - start_time
        timetables_pre = pd.concat(generated_timetables, ignore_index=True)

        self.timetables = post_process_timetable(self.timetables_pre, lines_info)
        self.timetables_path = f"initializer/files/{self.instance_name}.csv"
        self.timetables.to_csv(self.timetables_path)
        self.timetables['covered'] = False
        
        return self.timetables, self.timetables_path, self.timetables_generation_time
    
    def log_instance_generation(self, experiment_log_df, experiment_name):
        
        metrics = ["instance_generation_time", "number_of_trips", "number_of_lines", "instance_path"]
        log_ids = [f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}{i}" for i in range(len(metrics))]
        values = [self.timetables_generation_time, len(self.timetables), len(self.lines_info), self.timetables_path]
        df = pd.DataFrame({
            "log_ids": log_ids,
            "experiment_name": [experiment_name] * len(metrics),
            "step": ["timetables_generation"] * len(metrics),
            "metric": metrics,
            "value": values,
        })

        df = pd.concat([experiment_log_df, df], ignore_index=True)
        return df
        

class SolutionGenerator:
    def __init__(self, lines_info, cp_depot_distances, depots, timetables_path_to_use=None, seed=None, tmax=15.0, random_start_trip="no", random_tmax=False, tmax_min=0.0, inverse=False, experiment_name=1):
        self.lines_info = lines_info
        self.cp_depot_distances = cp_depot_distances
        self.cp_locations_summary = summarize_cp_locations(lines_info)
        self.cp_distances = calculate_cp_distances(self.cp_locations_summary, lines_info)
        self.transformed_cp_depot_distances = transform_cp_depot_distances(cp_depot_distances)
        self.dh_dict = merge_distances_dicts(self.transformed_cp_depot_distances, self.cp_distances)
        self.dh_df = make_deadhead_df(self.dh_dict)
        self.dh_times_df = make_deadhead_times_df(20, self.dh_df)
        self.seed = seed
        self.depots = depots
        self.instance_name = timetables_path_to_use.split("/")[-1].split(".")[0]
        self.timetables_path = timetables_path_to_use
        self.timetables = pd.read_csv(timetables_path_to_use, converters={"departure_time": pd.to_datetime})
        self.timetables['covered'] = False
        self.ti = None
        self.dist = 0
        self.time = 0
        self.route_total_dist = 0
        self.route_total_time = 0
        self.time_since_recharge = 0
        self.max_time = 16*60 # minutes
        self.max_dist = 120 # km
        self.recharging_rate = 30 # kw/hour
        self.consumption_rate = 15.6 # kw/hour
        self.route = []
        self.initial_solution = {}
        self.total_dh_dist = 0 # km
        self.total_dh_time = 0 # minutes
        self.total_travel_dist = 0 # km
        self.total_travel_time = 0 # minutes
        self.total_wait_time = 0 # minutes
        self.total_charging_time = 0 # minutes
        self.used_depots = []
        self.tmax = tmax
        self.random_start_trip=random_start_trip
        self.random_tmax=random_tmax
        self.tmax_min=tmax_min
        self.inverse=inverse
        self.experiment_name=experiment_name

    def generate_initial_set(self):
        start_time = time.time()
        i = 0
        while not self.timetables[self.timetables.covered == False].empty:
            if len(str(i)) == 1:
                comp = "00"
            elif len(str(i)) == 2:
                comp = "0"
            else:
                comp = ""
            try:
                ti = self.initialize(i)
                if ti is None:
                    break
            except:
                end_time = time.time()
                self.solution_generation_time = end_time - start_time
                return self.initial_solution, self.used_depots, self.instance_name, self.solution_generation_time
            continue_loop = True
            _i = 1
            while continue_loop:

                assert self.time <= self.max_time, f"Time limit exceeded! Route time is {self.time}"
                
                ti, continue_loop = self.update_route(ti)
                _i += 1

            recharging_freq = len(list(filter(lambda x: x[0] == "c", self.route)))

            route_cost = calculate_cost(
                recharging_freq, self.total_travel_time, self.total_dh_time, self.total_wait_time - self.total_charging_time) #MODIFIED
            
            self.initial_solution[f"Route_{i}"] = {
                "Path": self.route,
                "Cost": route_cost,
                "Data": {
                    "total_dh_dist": self.total_dh_dist,
                    "total_dh_time": self.total_dh_time,
                    "total_travel_dist": self.total_travel_dist,
                    "total_travel_time": self.total_travel_time,
                    "total_wait_time": self.total_wait_time,
                    "final_time": self.current_time,
                    "total_charging_time": self.total_charging_time
                }
            }

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

        end_time = time.time()
        self.solution_generation_time = end_time - start_time
        self.timetables.to_csv("initializer/files/timetables_covered.csv", index=False)
        pd.DataFrame(self.depots).to_csv(f"initializer/files/depots_initial_picture_{self.instance_name}.csv")

        metrics = [
            "solution_generation_time",
            "total_generated_columns"
        ]
    
        log_ids = [f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}{i}" for i in range(len(metrics))]
        values = [self.solution_generation_time, len(self.initial_solution)]
        log_df = pd.DataFrame({
            "log_ids": log_ids,
            "experiment_name": [self.experiment_name] * len(metrics),
            "step": ["initial_solution"] * len(metrics),
            "metric": metrics,
            "value": values,
        })


        return self.initial_solution, self.used_depots, self.instance_name, self.solution_generation_time, log_df


    def refresh_data(self):
        nodes = self.route[-2:]
        if nodes[0][0] == "d" and nodes[1][0] == "l":
            ti = self.timetables[self.timetables.trip_id == nodes[1]].iloc[0]
            dh_dist_ij = self.dh_df.loc[nodes[0], ti.start_cp_id]
            dh_time_ij = self.dh_times_df.loc[nodes[0], ti.start_cp_id]
            travel_dist = self.dh_df.loc[ti.start_cp_id, ti.dest_cp_id]
            travel_time = ti.planned_travel_time
            wait_time = 0
            equalizer = 0
            charging_time = 0
            self.current_time = ti.departure_time + timedelta(minutes=travel_time) 
        elif (nodes[0][0] == "l" or nodes[0][0] == "c") and nodes[1][0] == "l":
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
            time_diff = tj.departure_time - arrival_time_at_tj
            wait_time = time_diff.total_seconds() / 60
            equalizer = 0
            charging_time = 0

            self.current_time = tj.departure_time + timedelta(minutes=travel_time) 

        elif nodes[0][0] == "l" and nodes[1][0] == "c":
            dh_dist_ij = 0
            dh_time_ij = 0
            travel_dist = 0
            travel_time = 0
            
            charging_time = (self.consumption_rate * (self.time_since_recharge / 60) / self.recharging_rate) * 60
            wait_time = charging_time

            equalizer = self.time_since_recharge

            self.current_time += timedelta(minutes = dh_time_ij + travel_time + wait_time)

        else: #nodes[0][0] == "l" and nodes[1][0] == "d":
            tj = self.timetables[self.timetables.trip_id == nodes[0]].iloc[0]
            dh_dist_ij = self.dh_df.loc[tj.dest_cp_id, nodes[1]]
            dh_time_ij = self.dh_times_df.loc[tj.dest_cp_id, nodes[1]]
            travel_dist = 0
            travel_time = 0
            wait_time = 0
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

        if self.random_start_trip == "random_earliest":
            if (ti := get_earliest_trip_random(self.timetables)) is None:
                self.route.append(self.route[0])
                self.refresh_data()
                return None
        elif self.random_start_trip == "random":
            if (ti := get_random_start_trip(self.timetables)) is None:
                self.route_append(self.route[0])
                self.refresh_data()
                return None
        else:
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
        
        if self.random_tmax:
            if ti is None:
                ti = self.timetables[self.timetables.trip_id == self.route[-1]].iloc[0]
                tj = random_tmax_select_next_trip(eti=self.current_time, timetables=self.timetables, dh_times_df=self.dh_times_df, seed=self.seed, last_trip_id=ti.trip_id, ti=None, tmax=self.tmax, tmax_min=self.tmax_min, inverse=self.inverse)
            else:
                tj = random_tmax_select_next_trip(eti=self.current_time, timetables=self.timetables, dh_times_df=self.dh_times_df, seed=self.seed, last_trip_id=None, ti=ti, tmax=self.tmax, tmax_min=self.tmax_min, inverse=self.inverse)
        else:
            if ti is None:
                ti = self.timetables[self.timetables.trip_id == self.route[-1]].iloc[0]
                tj = select_next_trip(eti=self.current_time, timetables=self.timetables, dh_times_df=self.dh_times_df, seed=self.seed, last_trip_id=ti.trip_id, ti=None, tmax=self.tmax, tmax_min=self.tmax_min, inverse=self.inverse)
            else:
                tj = select_next_trip(eti=self.current_time, timetables=self.timetables, dh_times_df=self.dh_times_df, seed=self.seed, last_trip_id=None, ti=ti, tmax=self.tmax, tmax_min=self.tmax_min, inverse=self.inverse)

        if tj is None:
            self.route.append(self.route[0])
            self.refresh_data()
            return None, False
        
        tij_dist = self.dh_df.loc[ti.dest_cp_id, tj.start_cp_id]
        tij_time = self.dh_times_df.loc[ti.dest_cp_id, tj.start_cp_id]
        tj_dist = self.dh_df.loc[tj.start_cp_id, tj.dest_cp_id]

        pred_dist = self.dist + tij_dist + tj_dist
        time_diff = tj.departure_time - (self.current_time + timedelta(minutes=tij_time))
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

            charging_time = (self.consumption_rate * (self.time_since_recharge / 60) / self.recharging_rate) * 60

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

            #tj = select_next_trip(pred_current_time, self.timetables, self.dh_times_df, None, ti)

            uncovered = self.timetables[self.timetables.covered == False].sort_values(by='departure_time', ascending=True)
            dh_times = list(map(lambda x: self.dh_times_df.loc[ti.dest_cp_id, x], uncovered.start_cp_id))
            uncovered['tj_cp_arrivals'] = list(map(lambda x: ti.departure_time + timedelta(minutes=x), dh_times))
            uncovered['reachable'] = uncovered.departure_time >= uncovered.tj_cp_arrivals
            uncovered = uncovered[uncovered.reachable == True]
            try:
                tj = uncovered.iloc[0]
            except:
                tj = None

            if tj is None:
                self.route.append(self.route[0])
                self.refresh_data()
                return None, False
            
            tij_time = self.dh_times_df.loc[ti.dest_cp_id, tj.start_cp_id]
            time_diff = tj.departure_time - (self.current_time + timedelta(minutes=(charging_time + tij_time)))
            time_diff = time_diff.total_seconds() / 60 
            pred_time = self.time + charging_time + tij_time + time_diff + tj.planned_travel_time

            if pred_time >= self.max_time:
                #ESSE CHECKING NAO ACONTECE NO ALGORITMO OFICIAL!            
                self.route.append(self.route[0])
                self.refresh_data()
                return None, False
            
            # else:

            self.route.append(charging_station)
            self.refresh_data()
            self.dist = 0
            self.route.append(tj.trip_id)
            self.refresh_data()
            self.timetables.loc[self.timetables.trip_id == tj.trip_id, 'covered'] = True
        
            ## NÃO DELETAR A LINHA ABAIXO!!!
            ## OBS: step 2.2 inconsistente, por isso foi modificado. Não tem re-checking de Time. Procura em todas as timetables, quando poderia procurar primeiro no cp onde está.

            return None, True
        else:
            self.route.append(self.route[0])
            self.refresh_data()
            return None, False
        
    def log_solution_generation(self, experiment_log_df, experiment_name):
        
        metrics = ["solution_generation_time", "number_of_columns", "tmax"]
        log_ids = [f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}{i}" for i in range(len(metrics))]
        values = [self.solution_generation_time, len(self.initial_solution), self.tmax]
        df = pd.DataFrame({
            "log_ids": log_ids,
            "experiment_name": [experiment_name] * len(metrics),
            "step": ["solution_generation"] * len(metrics),
            "metric": metrics,
            "value": values,
        })

        df = pd.concat([experiment_log_df, df], ignore_index=True)
        return df

    def validate_initial_solution(self):
        """
        Validates the initial solution checking for:
        1. Repeated trips
        2. All trips covered
        3. All routes start and end at the same depot
        4. Route duration doesn't exceed max_time
        """
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        # 1. Check for repeated trips
        all_trips = []
        for route_name, route_data in self.initial_solution.items():
            route = route_data["Path"]
            # Extract only trip IDs (those starting with 'l')
            trips_in_route = [node for node in route if node.startswith('l')]
            all_trips.extend(trips_in_route)
        
        # Find duplicates
        seen_trips = set()
        duplicated_trips = set()
        for trip in all_trips:
            if trip in seen_trips:
                duplicated_trips.add(trip)
            seen_trips.add(trip)
        
        if duplicated_trips:
            validation_results["is_valid"] = False
            validation_results["errors"].append(f"Repeated trips found: {duplicated_trips}")
        
        # 2. Check if all trips were covered
        # Read the original timetables to get all trip IDs
        original_timetables = pd.read_csv(self.timetables_path)
        all_trip_ids = set(original_timetables['trip_id'].tolist())
        covered_trip_ids = set(all_trips)
        uncovered_trips = all_trip_ids - covered_trip_ids
        
        if uncovered_trips:
            validation_results["is_valid"] = False
            validation_results["errors"].append(f"Uncovered trips: {uncovered_trips}")
        
        # 3. Check if all routes start and end at the same depot
        for route_name, route_data in self.initial_solution.items():
            route = route_data["Path"]
            
            # Get first and last nodes
            if len(route) < 2:
                validation_results["is_valid"] = False
                validation_results["errors"].append(f"{route_name}: Route too short")
                continue
                
            first_node = route[0]
            last_node = route[-1]
            
            # Check if both are depots
            if not first_node.startswith('d'):
                validation_results["is_valid"] = False
                validation_results["errors"].append(f"{route_name}: Does not start with a depot (starts with {first_node})")
            
            if not last_node.startswith('d'):
                validation_results["is_valid"] = False
                validation_results["errors"].append(f"{route_name}: Does not end with a depot (ends with {last_node})")
            
            # Check if they are the same depot
            if first_node != last_node:
                validation_results["is_valid"] = False
                validation_results["errors"].append(f"{route_name}: Start depot ({first_node}) != End depot ({last_node})")
        
        # 4. Check if route duration exceeds max_time
        for route_name, route_data in self.initial_solution.items():
            route = route_data["Path"]
            
            # Find first trip in the route
            first_trip_id = None
            for node in route:
                if node.startswith('l'):
                    first_trip_id = node
                    break
            
            # Find last trip in the route (excluding the final depot)
            last_trip_id = None
            for node in reversed(route[:-1]):  # Exclude the final depot
                if node.startswith('l'):
                    last_trip_id = node
                    break
            
            if first_trip_id and last_trip_id:
                # Get departure time of first trip
                first_trip_data = original_timetables[original_timetables['trip_id'] == first_trip_id]
                if not first_trip_data.empty:
                    first_departure_time = pd.to_datetime(first_trip_data.iloc[0]['departure_time'])
                    
                    # Get end time of last trip (departure + travel time)
                    last_trip_data = original_timetables[original_timetables['trip_id'] == last_trip_id]
                    if not last_trip_data.empty:
                        last_departure_time = pd.to_datetime(last_trip_data.iloc[0]['departure_time'])
                        last_travel_time = last_trip_data.iloc[0]['planned_travel_time']
                        last_end_time = last_departure_time + timedelta(minutes=last_travel_time)
                        
                        # Calculate route duration
                        route_duration_minutes = (last_end_time - first_departure_time).total_seconds() / 60
                        
                        # Add time to return to depot if available
                        if 'Data' in route_data and 'final_time' in route_data['Data']:
                            # Use the final_time which includes the return to depot
                            final_time = route_data['Data']['final_time']
                            if isinstance(final_time, str):
                                final_time = pd.to_datetime(final_time)
                            route_duration_minutes = (final_time - first_departure_time).total_seconds() / 60
                        
                        if route_duration_minutes > self.max_time:
                            validation_results["is_valid"] = False
                            validation_results["errors"].append(
                                f"{route_name}: Duration ({route_duration_minutes:.2f} min) exceeds max_time ({self.max_time} min)"
                            )
        
        return validation_results