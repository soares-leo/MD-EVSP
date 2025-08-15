from datetime import datetime, timedelta
import datetime
import random
import pandas as pd
import math
from typing import Optional
import numpy as np
from scipy import stats

instances_discovery = {
    "59": {
        "gantt_chart_color": "orange",
        "start_time": "6:30",
        "end_time": "19:50"
    },
    "4": {
        "gantt_chart_color": "red",
        "start_time": "5:30",
        "end_time": "22:50"
    },
    "60": {
        "gantt_chart_color": "purple",
        "start_time": "6:00",
        "end_time": "21:10"
    }
}

def generate_timetable_v2(lines_info, line, total_trips, first_start_time="05:50", last_start_time="22:50"):

    trips = []
    trips_datetimes = [] # This will now store datetime.datetime objects
    trips_floats = []

    # FT1 = stats.Normal(mu=7, sigma=2)

    # FT2 = stats.Normal(mu=12, sigma=1.5)

    # FT3 = stats.Normal(mu=18, sigma=3)

    # FT_mixture = stats.Mixture([FT1, FT2, FT3], weights=[4.5/10.5, 1/10.5, 5/10.5])

    FT1 = stats.Normal(mu=7.5, sigma=1.5)
    FT2 = stats.Normal(mu=12, sigma=0.5)
    FT3 = stats.Normal(mu=13, sigma=0.5)
    FT4 = stats.Normal(mu=18.5, sigma=2)
    FT_mixture = stats.Mixture([FT1, FT2, FT3, FT4], weights=[0.4, 0.1, 0.05, 0.45])

    def time_to_float(time_str):
        hours, minutes = map(int, time_str.split(':'))
        return hours + minutes / 60

    first_start_time_float = time_to_float(first_start_time)
    last_start_time_float = time_to_float(last_start_time)

    # Use a fixed, dummy date for all operations
    dummy_date = datetime.date.today()

    trips_datetimes.append(
        datetime.datetime.combine(
            dummy_date,
            datetime.time(
                hour=int(first_start_time_float),
                minute= int((first_start_time_float - int(first_start_time_float)) * 60)
                )
            )
        )

    trips_datetimes.append(
        datetime.datetime.combine(
            dummy_date,
            datetime.time(
                hour=int(last_start_time_float),
                minute= int((last_start_time_float - int(last_start_time_float)) * 60)
                )
            )
        )

    trips.append(datetime.datetime.combine(dummy_date, datetime.time(hour=int(first_start_time.split(":")[0]), minute=int(first_start_time.split(":")[1]))).strftime('%H:%M'))
    trips.append(datetime.datetime.combine(dummy_date, datetime.time(hour=int(last_start_time.split(":")[0]), minute=int(last_start_time.split(":")[1]))).strftime('%H:%M'))

    trips_floats.append(first_start_time_float)
    trips_floats.append(last_start_time_float)

    i = 2
    while i < total_trips:

        u = np.random.uniform()
        st = FT_mixture.icdf(u)

        if st < first_start_time_float or st > last_start_time_float:
            continue

        hours = int(st)
        minutes = int((st - hours) * 60)

        if not (0 <= hours < 24 and 0 <= minutes < 60):
            continue

        current_datetime = datetime.datetime.combine(dummy_date, datetime.time(hour=hours, minute=minutes))

        # 2. Perform timedelta arithmetic on the full datetime object.
        time_window_start = current_datetime - timedelta(minutes=2)
        time_window_end = current_datetime + timedelta(minutes=2)

        # 3. Check if any existing trip falls within the +/- 5 minute window.
        #    This now correctly compares datetime objects with each other.
        check = list(filter(lambda dt: time_window_start < dt < time_window_end, trips_datetimes))

        if len(check) > 0:
            print()
            print(f"Skipping generated trip {current_datetime} because of minimal intevals:")
            print(check)
            print(f"Trips list currently with {len(trips)} trips.")
            print()
            continue

        # Get the time string for the final list
        time_str = current_datetime.strftime('%H:%M')

        if time_str in trips:
            print()
            print("Skipping generated trip because it is already in the list:")
            print(time_str)
            print(f"Trips list currently with {len(trips)} trips.")
            print()
            continue

        print()
        print(f"Generated trip at {time_str}")
        trips.append(time_str)
        print(f"Trips list currently with {len(trips)} trips.")
        print()
        # 4. Store the full datetime object for future comparisons.
        trips_datetimes.append(current_datetime)
        trips_floats.append(st)
        i += 1

    pdfs = [FT_mixture.pdf(t) for t in trips_floats]

    speeds = [20 * (1+ (-0.5 * ((FT_mixture.pdf(t) - min(pdfs)) / (max(pdfs) - min(pdfs))) + 0.25)) for t in trips_floats]

    start_cps = []
    dest_cps = []
    """
    cps_tuples = []

    left = 0
    right = len(trips) - 1

    while len(cps_tuples) < len(trips):
        cps_tuples.extend([(left, "cp1"), (right, "cp1")])
        left+=1
        right-=1
        if len(cps_tuples) == len(trips):
            break
        cps_tuples.extend([(left, "cp2"), (right, "cp2")])
        left+=1
        right-=1

    cps_tuples = sorted(cps_tuples, key=lambda x: x[0])
    """

    start_cps = ["cp1", "cp2"] * (len(trips) // 2)

    dest_cps = ["cp2" if x == "cp1" else "cp1" for x in start_cps]

    start_cp_ids = [f"{x}_l{line}" for x in start_cps]
    dest_cp_ids = [f"{x}_l{line}" for x in dest_cps]

    trip_ids = [f"l{line}_{x}_{i+1}" for i, x in enumerate(start_cps)]

    print("---------------------------------------------------------------------------------------")
    print(len(trip_ids))

    pdfs = [FT_mixture.pdf(t) for t in trips_floats]

    departure_times_str = [time_point.strftime("%H:%M") for time_point in trips_datetimes]

    import pandas as pd

    numbers_df = pd.DataFrame({
        "departure_time": trips_datetimes,
        "departure_time_str": departure_times_str,
        "datetime_obj": trips_datetimes,
        "datetime_float": trips_floats,
        "pdf": pdfs,
        "speed": speeds
    })

    numbers_df.sort_values(by="departure_time", ascending=True, ignore_index=True, inplace=True)

    print("trip_id:", len(trip_ids))
    print("start_cp:", len(start_cps))
    print("start_cp_id:", len(start_cp_ids))
    print("dest_cp:", len(dest_cps))
    print("dest_cp_id:", len(dest_cp_ids))
    
    names_df = pd.DataFrame({
        "trip_id": trip_ids,
        "line": [line] * len(trip_ids),
        "start_cp": start_cps,
        "start_cp_id": start_cp_ids,
        "dest_cp": dest_cps,
        "dest_cp_id": dest_cp_ids
    })

    final_df = pd.concat([names_df, numbers_df], axis=1)

    final_df.sort_values(by=["start_cp","departure_time"], ascending=True, ignore_index=True, inplace=True)

    intervals = []
    travel_times = []
    last_start_cp = None

    for index, row in final_df.iterrows():
        current_start_cp = row["start_cp"]
        if current_start_cp != last_start_cp: 
            last_start_cp = current_start_cp
            departure_interval = 0
        else:
            departure_interval = (row["departure_time"] - final_df.loc[index-1, "departure_time"]).total_seconds() / 60
        distance = lines_info[str(line)]["cp_distance_km"]
        planned_travel_time = float(round((distance / row["speed"]) * 60))
        intervals.append(departure_interval)
        travel_times.append(planned_travel_time)

    final_df["departure_interval"] = intervals
    final_df["planned_travel_time"] = travel_times

    return final_df


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
    seed: Optional[int] = None,
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
    tmax = 240.0 #timetables[(timetables.line == ti.line) & (timetables.covered == False)]["departure_interval"].max()
    # try:
    #     int(tmax)
    # except:
    #     tmax = 0
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

        tj = tj_candidates.sample(random_state=seed)

    except:

        other_lines = list(timetables[timetables.covered == False]['line'].unique())
        if ti_line in other_lines:
            other_lines.remove(ti_line)
        #tmax_per_line = {l: find_tmax(l) for l in other_lines}
        
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

        tj_pre_candidates['tmax'] = [timedelta(minutes=180.01)] * len(tj_pre_candidates) #tj_pre_candidates.apply(
        #    lambda row: timedelta(minutes=tmax_per_line[row['line']]), axis=1
        #)

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
    
    return depot_routes, routes_costs