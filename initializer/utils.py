from datetime import datetime, timedelta
import random
import pandas as pd
import math
from typing import Optional

def generate_timetable(line, total_trips, start_time="05:50", min_interval=5, max_interval=15,
                       interval_bias=8, travel_time_range=(70, 75)):

    current_time = datetime.strptime(start_time, "%H:%M")
    trips = []
    int_div = total_trips // 2
    mod = total_trips % 2
    num_trips_tuple = (int_div, int_div + mod)
    cps = ("cp1", "cp2")
    start_times_tuple = (current_time, current_time + timedelta(minutes=10))

    for num_trips, cp, time, idx in zip(num_trips_tuple, cps, start_times_tuple, (1,0)):
        for i in range(1, num_trips + 1):
            if i == 1:
                interval = None
            else:
                interval = int(random.gauss(interval_bias, 2))
                interval = max(min_interval, min(interval, max_interval))
                time += timedelta(minutes=interval)

            travel_time = float(random.randint(*travel_time_range))

            trips.append({
                'trip_id': f"l{line}_{cp}_{i}",
                'line': line,
                'start_cp': cp,
                'start_cp_id': f"{cp}_l{line}",
                'dest_cp': cps[idx],
                'dest_cp_id': f"{cps[idx]}_l{line}",
                'departure_time': time,
                'departure_interval': interval,
                'planned_travel_time': travel_time
            })

    return pd.DataFrame(trips)

def haversine_distance(latlon1, latlon2):

    R = 6371

    lat1, lon1 = latlon1[1], latlon1[0]
    lat2, lon2 = latlon2[1], latlon2[0]

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c * 1.2

    return distance

def summarize_cp_locations(lines_info):
    cp_locations_summary = {}
    for line_id, line_data in lines_info.items():
        for cp in list(line_data.keys())[:2]:
            cp_id = f"{cp}_l{line_id}"
            latlon = line_data[cp]["latlon"]
            cp_locations_summary[cp_id] = latlon
    return cp_locations_summary

def calculate_cp_distances(cp_locations_summary, lines_info):
    from_distances = {}

    # Criar um mapeamento auxiliar: cp_id -> line
    cp_to_line = {}
    for line_id, data in lines_info.items():
        for cp_key in ["cp1", "cp2"]:
            cp_id = f"{cp_key}_l{line_id}"
            cp_to_line[cp_id] = line_id

    for from_cp, from_coords in cp_locations_summary.items():
        to_distances = {}
        for to_cp, to_coords in cp_locations_summary.items():
            if from_cp == to_cp:
                to_distances[to_cp] = 0.0
                continue

            from_line = cp_to_line.get(from_cp)
            to_line = cp_to_line.get(to_cp)

            # Se são da mesma linha e são pares opostos, usa a distância fornecida
            if from_line == to_line:
                cp_keys = [k.split("_")[0] for k in (from_cp, to_cp)]
                if set(cp_keys) == {"cp1", "cp2"}:
                    to_distances[to_cp] = lines_info[from_line]["cp_distance_km"]
                    continue

            # Caso contrário, calcula com Haversine
            to_distances[to_cp] = haversine_distance(from_coords, to_coords)

        from_distances[from_cp] = to_distances

    return from_distances

def transform_cp_depot_distances(original_dict):
    inverse_dict = {}

    for cp, depot_distances in original_dict.items():
        for depot, distance in depot_distances.items():
            if depot not in inverse_dict:
                inverse_dict[depot] = {}
            inverse_dict[depot][cp] = distance

    combined_dict = {**original_dict, **inverse_dict}

    return combined_dict

def merge_distances_dicts(dict_1, dict_2):
    merged_dict = {}

    for key, distances in dict_1.items():
        merged_dict[key] = distances.copy()
        if key in dict_2:
            for subkey, fallback_distance in dict_2[key].items():
                if subkey not in merged_dict[key]:
                    merged_dict[key][subkey] = fallback_distance

    return merged_dict

def make_deadhead_df(deadhead_dict):
    sorted_dict = {k:v for k, v in sorted(deadhead_dict.items(), key=lambda item: item[0])}
    for k in list(sorted_dict.keys()):
        sorted_subdict = {_k: _v for _k, _v in sorted(sorted_dict[k].items(), key=lambda item: item[0])}
        sorted_dict[k] = sorted_subdict
    deadhead_df = pd.DataFrame(sorted_dict)
    return deadhead_df

def make_deadhead_times_df(avg_speed_kmh, deadhead_df):
    conversor = lambda x: x/avg_speed_kmh*60
    deadhead_times_df = deadhead_df.apply(conversor)
    return deadhead_times_df

def get_earliest_trip(timetables: pd.DataFrame) -> dict:
    
    uncovered = timetables[timetables['covered'] == False]
    
    if uncovered.empty:
        result = None
    else:
        min_departure_time = uncovered['departure_time'].min()
        earliest_uncovered = uncovered[uncovered['departure_time'] == min_departure_time]
        if earliest_uncovered.empty:
            result = None
        else:
            result = earliest_uncovered.iloc[0]
    
    return result

def get_nearest_depot(cp_id: str, depots: dict, cp_depot_distances: dict) -> str:
    for k in cp_depot_distances[cp_id].keys():
        if depots[k]['departed'] == depots[k]['capacity']:
            continue
        else:
            return k
    print("No depots found!")
    return None
def select_next_trip(
    eti,
    timetables: pd.DataFrame,
    dh_times_df: pd.DataFrame,
    last_trip_id: Optional[str] = None,
    ti: Optional[pd.Series] = None
) -> str:
    if (last_trip_id is None) == (ti is None):
        raise ValueError("Exactly one of last_trip_id or trip_object must be provided.")
    
    if last_trip_id is not None:
        ti = timetables[timetables.trip_id == last_trip_id].iloc[0]    
    def find_tmax(line):
        df = timetables[(timetables.line == line) & (timetables.covered == False)]['departure_interval']
        return df.max()
    
    
    ti_line = ti.line    
    tmax = timetables[(timetables.line == ti.line) & (timetables.covered == False)]["departure_interval"].max()
    try:
        int(tmax)
    except:
        tmax = 0
    tj_start_cp = ti.dest_cp_id
    tj_candidates = timetables[
        (
            (timetables.start_cp_id == tj_start_cp)
                & (timetables.covered == False)
                & (timetables.departure_time >= eti)
                & (timetables.departure_time <= eti + timedelta(minutes=tmax))
        )
    ].copy()
    # print()
    # print(tj_candidates)
    try:

        tj = tj_candidates.sample()

    except:

        other_lines = list(timetables[timetables.covered == False]['line'].unique())
        if ti_line in other_lines:
            other_lines.remove(ti_line)
        tmax_per_line = {l: find_tmax(l) for l in other_lines}
        
        tj_pre_candidates = timetables[
            (
                (timetables.line.isin(other_lines))
                    & (timetables.covered == False)
            )
        ].copy()

        if tj_pre_candidates.empty:
            return None
        
        tj_pre_candidates['dh_time'] = tj_pre_candidates.apply(
            lambda row: timedelta(minutes=dh_times_df.loc[tj_start_cp, row['start_cp_id']]), axis=1
        )

        tj_pre_candidates['lower_bound'] = eti + tj_pre_candidates['dh_time']

        tj_pre_candidates['tmax'] = tj_pre_candidates.apply(
            lambda row: timedelta(minutes=tmax_per_line[row['line']]), axis=1
        )

        tj_pre_candidates['upper_bound'] = (
            eti
                + tj_pre_candidates['dh_time']
                + tj_pre_candidates['tmax']
        )

        tj_candidates = tj_pre_candidates[
            (
                (tj_pre_candidates.departure_time <= tj_pre_candidates['upper_bound'])
                    & (tj_pre_candidates.departure_time >= tj_pre_candidates['lower_bound'])
            )
        ]

        if tj_candidates.empty:
            return None

        tj = tj_candidates[tj_candidates.dh_time == tj_candidates.dh_time.min()]

    return tj.iloc[0]

def calculate_cost(recharging_freq, travel_time, dh_time, waiting_time):
    vehicle_fixed_cost = 700
    charging_cost = 35 * recharging_freq
    
    travel_cost = 1.6 * travel_time
    dh_cost = 1.6 * dh_time
    waiting_cost = 0.7 * waiting_time

    operational_cost = travel_cost + dh_cost + waiting_cost

    total_cost = vehicle_fixed_cost + charging_cost + operational_cost

    return total_cost

def group_routes_by_depot(initial_solution):
    depot_routes = {}
    routes_costs = {}

    for route_name, route_data in initial_solution.items():
        route = route_data["Path"]
        cost = route_data["Cost"]
        
        starting_depot = route[0]
        
        if not depot_routes.get(starting_depot):
            depot_routes[starting_depot] = []
            routes_costs[starting_depot] = []

        depot_routes[starting_depot].append(route)
        routes_costs[starting_depot].append(cost)

    for k, v in depot_routes.items():
        print(f"Number of vehicles departed from {k}: {len(v)}")
    
    return depot_routes, routes_costs