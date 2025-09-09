from datetime import datetime, timedelta
import datetime
import random
import pandas as pd
import math
from typing import Optional
import numpy as np
from scipy import stats

def generate_timetable_v3(lines_info, line, total_trips, first_start_time="05:50", last_start_time="22:35"):
    """
    Generate timetable that better matches the real-world problem instance.
    Uses a hybrid approach: deterministic for regular periods, stochastic for peak hours.
    """
    
    trips_datetimes = []
    dummy_date = datetime.date.today()
    
    def time_str_to_datetime(time_str):
        hours, minutes = map(int, time_str.split(':'))
        return datetime.datetime.combine(dummy_date, datetime.time(hour=hours, minute=minutes))
    
    def datetime_to_float(dt):
        return dt.hour + dt.minute / 60
    
    # Define time periods and their characteristics
    time_periods = {
        "early_morning": {
            "start": time_str_to_datetime("05:50"),
            "end": time_str_to_datetime("07:00"),
            "interval_mean": 7.5,  # 7-8 minute intervals as shown in Table 3
            "interval_std": 0.5,
            "weight": 0.08  # Lower frequency
        },
        "morning_peak": {
            "start": time_str_to_datetime("07:00"),
            "end": time_str_to_datetime("09:30"),
            "interval_mean": 3.5,  # Much denser during peak
            "interval_std": 1.0,
            "weight": 0.25  # High frequency
        },
        "morning_shoulder": {
            "start": time_str_to_datetime("09:30"),
            "end": time_str_to_datetime("11:30"),
            "interval_mean": 5.0,
            "interval_std": 1.5,
            "weight": 0.12
        },
        "midday": {
            "start": time_str_to_datetime("11:30"),
            "end": time_str_to_datetime("14:00"),
            "interval_mean": 6.0,
            "interval_std": 2.0,
            "weight": 0.10
        },
        "afternoon": {
            "start": time_str_to_datetime("14:00"),
            "end": time_str_to_datetime("17:00"),
            "interval_mean": 4.5,
            "interval_std": 1.5,
            "weight": 0.15
        },
        "evening_peak": {
            "start": time_str_to_datetime("17:00"),
            "end": time_str_to_datetime("20:00"),
            "interval_mean": 3.0,  # Very dense during evening peak
            "interval_std": 1.0,
            "weight": 0.25  # High frequency
        },
        "late_evening": {
            "start": time_str_to_datetime("20:00"),
            "end": time_str_to_datetime("22:35"),
            "interval_mean": 12.5,  # 10-15 minute intervals as shown
            "interval_std": 2.5,
            "weight": 0.05  # Lower frequency
        }
    }
    
    # Generate trips for each period
    for period_name, period_info in time_periods.items():
        current_time = period_info["start"]
        period_end = period_info["end"]
        
        # Add first trip of the period
        if current_time == time_str_to_datetime("05:50"):
            trips_datetimes.append(current_time)
        
        while current_time < period_end and len(trips_datetimes) < total_trips:
            # Generate interval based on period characteristics
            interval = max(2, np.random.normal(period_info["interval_mean"], period_info["interval_std"]))
            interval = round(interval)  # Round to nearest minute
            
            next_time = current_time + timedelta(minutes=interval)
            
            # Check minimum spacing (relaxed to 1.5 minutes for peak periods)
            min_interval = 1.5 if "peak" in period_name else 2.0
            
            # Check if too close to existing trips
            too_close = False
            for existing_trip in trips_datetimes:
                if abs((next_time - existing_trip).total_seconds() / 60) < min_interval:
                    too_close = True
                    break
            
            if not too_close and next_time <= period_end:
                trips_datetimes.append(next_time)
                current_time = next_time
            else:
                current_time = current_time + timedelta(minutes=1)
    
    # Sort trips
    trips_datetimes.sort()
    
    # Trim to exact number of trips if we have too many
    if len(trips_datetimes) > total_trips:
        trips_datetimes = trips_datetimes[:total_trips]
    
    # Add more trips if needed (fill gaps)
    while len(trips_datetimes) < total_trips:
        # Find largest gap
        max_gap = 0
        max_gap_index = 0
        for i in range(len(trips_datetimes) - 1):
            gap = (trips_datetimes[i + 1] - trips_datetimes[i]).total_seconds() / 60
            if gap > max_gap:
                max_gap = gap
                max_gap_index = i
        
        # Insert trip in middle of largest gap
        if max_gap > 3:  # Only fill if gap is large enough
            new_time = trips_datetimes[max_gap_index] + (trips_datetimes[max_gap_index + 1] - trips_datetimes[max_gap_index]) / 2
            trips_datetimes.insert(max_gap_index + 1, new_time)
            trips_datetimes.sort()
        else:
            break
    
    # Calculate derived values
    trips_floats = [datetime_to_float(dt) for dt in trips_datetimes]
    departure_times_str = [dt.strftime("%H:%M") for dt in trips_datetimes]
    
    # Assign CPs (alternating pattern)
    start_cps = []
    for i in range(len(trips_datetimes)):
        start_cps.append("cp1" if i % 2 == 0 else "cp2")
    
    dest_cps = ["cp2" if x == "cp1" else "cp1" for x in start_cps]
    start_cp_ids = [f"{x}_l{line}" for x in start_cps]
    dest_cp_ids = [f"{x}_l{line}" for x in dest_cps]
    trip_ids = [f"l{line}_{start_cps[i]}_{i+1}" for i in range(len(trips_datetimes))]
    
    # Calculate speeds based on time of day (congestion patterns)
    speeds = []
    for dt in trips_datetimes:
        hour = dt.hour
        if 7 <= hour < 9 or 17 <= hour < 19:  # Peak hours - slower
            speed = np.random.normal(18, 2)
        elif 22 <= hour or hour < 6:  # Late night - faster
            speed = np.random.normal(25, 2)
        else:  # Normal hours
            speed = np.random.normal(20, 2)
        speeds.append(max(15, min(30, speed)))  # Clamp between 15-30 km/h
    
    # Create names dataframe
    names_df = pd.DataFrame({
        "trip_id": trip_ids,
        "line": [line] * len(trip_ids),
        "start_cp": start_cps,
        "start_cp_id": start_cp_ids,
        "dest_cp": dest_cps,
        "dest_cp_id": dest_cp_ids
    })
    
    # Create numbers dataframe
    numbers_df = pd.DataFrame({
        "departure_time": trips_datetimes,
        "departure_time_str": departure_times_str,
        "datetime_obj": trips_datetimes,
        "datetime_float": trips_floats,
        "pdf": [0] * len(trips_datetimes),  # Placeholder
        "speed": speeds
    })
    
    # Combine dataframes
    final_df = pd.concat([names_df, numbers_df], axis=1)
    final_df.sort_values(by=["start_cp", "departure_time"], ascending=True, ignore_index=True, inplace=True)
    
    # Calculate intervals and travel times
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
        # Travel time directly from speed - no need for additional adjustments
        planned_travel_time = float(round((distance / row["speed"]) * 60))
        
        intervals.append(departure_interval)
        travel_times.append(planned_travel_time)
    
    final_df["departure_interval"] = intervals
    final_df["planned_travel_time"] = travel_times
    
    # Print summary statistics for validation
    print(f"\nGenerated {len(final_df)} trips for line {line}")
    print(f"First trip: {final_df.iloc[0]['departure_time_str']}")
    print(f"Last trip: {final_df.iloc[-1]['departure_time_str']}")
    
    # Sample intervals and speeds by hour for validation
    print(f"\nHourly speed and travel time patterns:")
    df_by_hour = final_df.groupby(final_df['departure_time'].dt.hour).agg({
        'departure_interval': lambda x: x[x > 0].mean() if len(x[x > 0]) > 0 else 0,
        'speed': 'first',  # All trips in same hour have same speed
        'planned_travel_time': 'first'  # Should be consistent for same hour
    }).round(1)
    
    for hour, row in df_by_hour.iterrows():
        if row['departure_interval'] > 0:
            print(f"  Hour {hour:02d}: Speed={row['speed']:.1f} km/h, "
                  f"Travel time={row['planned_travel_time']:.0f} min, "
                  f"Avg interval={row['departure_interval']:.1f} min")
    
    return final_df

def generate_timetable_vX(lines_info, line, total_trips, first_start_time="05:50", last_start_time="22:50"):
    """
    Generate timetable using a stratified sampling approach to match a target distribution.
    This avoids the infinite loop issue from the previous rejection sampling method.
    """
    trips_datetimes = []
    
    # --- Setup remains the same ---
    FT1 = stats.Normal(mu=7, sigma=2)
    FT2 = stats.Normal(mu=12, sigma=2)
    FT4 = stats.Normal(mu=18, sigma=3)
    FT_mixture = stats.Mixture([FT1, FT2, FT4], weights=[0.35, 0.2, 0.45])

    def time_to_float(time_str):
        hours, minutes = map(int, time_str.split(':'))
        return hours + minutes / 60

    dummy_date = datetime.date.today()
    
    # Define time intervals and their theoretical proportions
    today_start_hour = datetime.datetime.combine(dummy_date, datetime.time(5, 0))
    hourly_dict_prop = {}
    for i in range((24 - 5) * 4):
        interval_start = today_start_hour + timedelta(minutes=i * 15)
        interval_end = interval_start + timedelta(minutes=15) # Non-overlapping
        
        float_clock_start = interval_start.hour + interval_start.minute / 60
        float_clock_end = interval_end.hour + interval_end.minute / 60
        
        # Ensure end time is handled correctly for CDF calculation
        if float_clock_end == 0.0: float_clock_end = 24.0

        hourly_dict_prop[(interval_start, interval_end)] = FT_mixture.cdf(float_clock_end) - FT_mixture.cdf(float_clock_start)
        
    # --- Add fixed first and last trips ---
    first_start_time_float = time_to_float(first_start_time)
    last_start_time_float = time_to_float(last_start_time)

    first_trip = datetime.datetime.combine(
        dummy_date,
        datetime.time(hour=int(first_start_time_float), minute=int((first_start_time_float - int(first_start_time_float)) * 60))
    )
    last_trip = datetime.datetime.combine(
        dummy_date,
        datetime.time(hour=int(last_start_time_float), minute=int((last_start_time_float - int(last_start_time_float)) * 60))
    )
    
    trips_datetimes.extend([first_trip, last_trip])
    
    # --- NEW: Stratified Sampling Logic ---
    trips_to_generate = total_trips - 2
    if trips_to_generate > 0:
        # 1. Calculate the ideal number of trips for each interval (can be float)
        target_counts_float = {
            interval: prop * trips_to_generate
            for interval, prop in hourly_dict_prop.items()
        }

        # 2. Round to get integer counts (quotas)
        target_counts_int = {
            interval: round(count)
            for interval, count in target_counts_float.items()
        }

        # 3. Correct for rounding errors to ensure the total is exact
        current_sum = sum(target_counts_int.values())
        difference = trips_to_generate - current_sum

        if difference != 0:
            # Sort intervals by their rounding error to make intelligent adjustments
            errors = {
                interval: target_counts_float[interval] - target_counts_int[interval]
                for interval in target_counts_int
            }
            # If we need more trips (diff > 0), add to intervals with highest positive error (e.g., 4.9 -> 5)
            # If we need fewer trips (diff < 0), subtract from intervals with lowest negative error (e.g., 4.1 -> 4)
            intervals_to_adjust = sorted(errors, key=errors.get, reverse=(difference > 0))
            
            for i in range(abs(int(difference))):
                interval_to_change = intervals_to_adjust[i]
                target_counts_int[interval_to_change] += 1 if difference > 0 else -1

        # 4. Generate trips within each interval according to its quota
        for interval, count in target_counts_int.items():
            start_dt, end_dt = interval
            start_ts = start_dt.timestamp()
            end_ts = end_dt.timestamp()

            for _ in range(int(count)):
                random_ts = random.uniform(start_ts, end_ts)
                new_trip_dt = datetime.datetime.fromtimestamp(random_ts)
                
                # Filter out trips outside the operational time window
                if first_trip <= new_trip_dt <= last_trip:
                    trips_datetimes.append(new_trip_dt)
    
    # --- Post-generation processing remains mostly the same ---
    
    # Remove duplicates and sort
    trips_datetimes = sorted(list(set(trips_datetimes)))

    # If we generated too many due to edge cases, trim the list.
    # A better approach would be to regenerate a few if too few, but trimming is simpler.
    if len(trips_datetimes) > total_trips:
        # This can happen if random times coincide with first/last trip.
        # We can randomly sample down to the desired number.
        trips_datetimes = sorted(random.sample(trips_datetimes, total_trips))
        
    # Convert to floats for PDF and speed calculations
    trips_floats = [(dt.hour + dt.minute / 60) for dt in trips_datetimes]
    
    pdfs = [FT_mixture.pdf(t) for t in trips_floats]
    min_pdf, max_pdf = min(pdfs), max(pdfs)
    # Avoid division by zero if all pdfs are the same
    pdf_range = max_pdf - min_pdf if max_pdf > min_pdf else 1 
    
    speeds = [20 * (1 + (-0.5 * ((pdf - min_pdf) / pdf_range) + 0.25)) for pdf in pdfs]

    # Assign CPs, IDs, etc.
    start_cps = ["cp1" if i % 2 == 0 else "cp2" for i in range(len(trips_datetimes))]
    dest_cps = ["cp2" if x == "cp1" else "cp1" for x in start_cps]
    start_cp_ids = [f"{x}_l{line}" for x in start_cps]
    dest_cp_ids = [f"{x}_l{line}" for x in dest_cps]
    trip_ids = [f"l{line}_{start_cps[i]}_{i+1}" for i in range(len(trips_datetimes))]
    departure_times_str = [dt.strftime("%H:%M") for dt in trips_datetimes]

    # --- Create DataFrame ---
    names_df = pd.DataFrame({
        "trip_id": trip_ids,
        "line": [line] * len(trip_ids),
        "start_cp": start_cps,
        "start_cp_id": start_cp_ids,
        "dest_cp": dest_cps,
        "dest_cp_id": dest_cp_ids
    })

    numbers_df = pd.DataFrame({
        "departure_time": trips_datetimes,
        "departure_time_str": departure_times_str,
        "datetime_obj": trips_datetimes,
        "datetime_float": trips_floats,
        "pdf": pdfs,
        "speed": speeds
    })

    final_df = pd.concat([names_df, numbers_df], axis=1)
    final_df.sort_values(by=["start_cp", "departure_time"], ascending=True, ignore_index=True, inplace=True)

    # --- Calculate intervals and travel times ---
    intervals = []
    travel_times = []
    last_start_cp = None
    
    # Use a temporary dict for last departure time per CP to handle sorted data
    last_departure_time = {}

    for index, row in final_df.iterrows():
        current_start_cp = row["start_cp"]
        
        if current_start_cp not in last_departure_time:
            departure_interval = 0
        else:
            departure_interval = (row["departure_time"] - last_departure_time[current_start_cp]).total_seconds() / 60
        
        last_departure_time[current_start_cp] = row["departure_time"]
        
        distance = lines_info[str(line)]["cp_distance_km"]
        planned_travel_time = float(round((distance / row["speed"]) * 60))
        
        intervals.append(departure_interval)
        travel_times.append(planned_travel_time)

    final_df["departure_interval"] = intervals
    final_df["planned_travel_time"] = travel_times

    return final_df

def generate_timetable_v2(lines_info, line, total_trips, first_start_time="05:50", last_start_time="22:50"):

    trips = []
    trips_datetimes = [] # This will now store datetime.datetime objects
    trips_floats = []



    FT1 = stats.Normal(mu=7, sigma=3)
    FT2 = stats.Normal(mu=12, sigma=3)
    FT4 = stats.Normal(mu=18, sigma=4)
    FT_mixture = stats.Mixture([FT1, FT2, FT4], weights=[0.4, 0.1, 0.5])

    def time_to_float(time_str):
        hours, minutes = map(int, time_str.split(':'))
        return hours + minutes / 60

    first_start_time_float = time_to_float(first_start_time)
    last_start_time_float = time_to_float(last_start_time)

    # Use a fixed, dummy date for all operations
    dummy_date = datetime.date.today()

    today = dummy_date
    today_start_hour = datetime.datetime.combine(today, datetime.time()) + timedelta(minutes=5*60)
    today_end_hour = datetime.datetime.combine(today, datetime.time()) + timedelta(minutes=23*60 + 59)
    print(today_start_hour)
    print(today_end_hour)
    hourly_dict = {(today_start_hour + timedelta(hours=i*0.2), today_start_hour + timedelta(hours=i*0.2) + timedelta(minutes=29)): [] for i in range((24-5)*5)}
    hourly_dict_prop = {}
    for k, v in hourly_dict.items():
        float_clock_start = k[0].hour + k[0].minute / 60
        float_clock_end = k[1].hour + k[1].minute / 60
        hourly_dict_prop[k] = FT_mixture.cdf(float_clock_end) - FT_mixture.cdf(float_clock_start)

    first_trip = datetime.datetime.combine(
        dummy_date,
        datetime.time(
            hour=int(first_start_time_float),
            minute= int((first_start_time_float - int(first_start_time_float)) * 60)
            )
        )
    
    last_trip = datetime.datetime.combine(
        dummy_date,
        datetime.time(
            hour=int(last_start_time_float),
            minute= int((last_start_time_float - int(last_start_time_float)) * 60)
            )
        )

    _, first_trip_match_interval = check_proportion(hourly_dict, hourly_dict_prop, first_trip, trips_datetimes)
    _, last_trip_match_interval = check_proportion(hourly_dict, hourly_dict_prop, last_trip, trips_datetimes)

    hourly_dict[first_trip_match_interval].append(1)
    hourly_dict[last_trip_match_interval].append(1)
    
    trips_datetimes.append(first_trip)

    trips_datetimes.append(last_trip)

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
        # time_window_start = current_datetime - timedelta(minutes=2)
        # time_window_end = current_datetime + timedelta(minutes=2)

        # max_time_window_start = current_datetime - timedelta(minutes=20)
        # max_time_window_end = current_datetime + timedelta(minutes=20)
        time_range_checks = []
        # 3. Check if any existing trip falls within the +/- 5 minute window.
        #    This now correctly compares datetime objects with each other.
        for dt in trips_datetimes:
            min_time_window_left = dt - timedelta(minutes=60)
            max_time_window_left = dt - timedelta(minutes=2)
            min_time_window_right = dt + timedelta(minutes=2)
            max_time_window_right = dt + timedelta(minutes=60)

            if (current_datetime > max_time_window_left and current_datetime < min_time_window_right): #or (current_datetime < min_time_window_left or current_datetime > max_time_window_right):
                time_range_checks.append((max_time_window_left.strftime('%Y-%m-%d%H:%M'), current_datetime.strftime('%Y-%m-%d%H:%M'), min_time_window_right.strftime('%Y-%m-%d%H:%M')))                        

        if len(time_range_checks) > 0:
            print()
            print(f"Skipping generated trip {current_datetime} because of minimal intevals:")
            print(time_range_checks)
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
    
        hourly_dict_copy = hourly_dict_prop.copy()

        prop_check, trip_match_interval = check_proportion(hourly_dict, hourly_dict_prop, current_datetime, trips_datetimes)

        if not prop_check:
            continue

        hourly_dict[trip_match_interval].append(1)


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

    speeds = [20 * (1+ (-0.5 * ((FT_mixture.pdf(t) - min(pdfs)) / (max(pdfs) - min(pdfs))) + (1/3))) for t in trips_floats]

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

def check_proportion(hourly_dict, hourly_dict_prop, trip_datetime_obj, trips_datetimes):
    hourly_dict_copy = hourly_dict.copy()
    while len(hourly_dict_copy) > 1:
        dict_len = len(hourly_dict_copy)
        half_rounded = dict_len // 2
        left_len = dict_len - half_rounded
        right_len = half_rounded
        left_dict = dict(list(hourly_dict_copy.items())[:left_len])
        right_dict = dict(list(hourly_dict_copy.items())[-right_len:])
        left_check_key = list(left_dict.keys())[-1]
        if trip_datetime_obj <= left_check_key[1]:
            hourly_dict_copy = left_dict
        else:
            hourly_dict_copy = right_dict
    
    mach_interval = list(hourly_dict_copy.keys())[0]
    if len(trips_datetimes) == 0:
        print("Proportion check: success.")
        return True, mach_interval
    trips_in_interval = len(hourly_dict[mach_interval])
    trips_proportion = trips_in_interval / len(trips_datetimes)
    if trips_proportion > hourly_dict_prop[mach_interval]:
        print("Proportion check: failed.")
        print(f"Expected proportion: {hourly_dict_prop[mach_interval]}")
        print(f"Actual proportion: {trips_proportion}")
        return False, None
    else:
        print("Proportion check: success.")
        return True, mach_interval

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

def get_random_start_trip(timetables: pd.DataFrame) -> dict:
    uncovered = timetables[timetables['covered'] == False]    
    if uncovered.empty:
        result = None
    else:
        result = uncovered.sample(n=1).iloc[0]
    return result

def get_earliest_trip_random(timetables: pd.DataFrame) -> dict:
    
    uncovered = timetables[timetables['covered'] == False]
    
    # Get the first 10 trips for each line
    earliest_per_line = []
    for line in uncovered['line'].unique():
        line_trips = uncovered[uncovered['line'] == line]
        line_trips_sorted = line_trips.sort_values(by='departure_time', ascending=True).head(10)
        earliest_per_line.append(line_trips_sorted)
    
    # Combine all the earliest trips from each line
    if earliest_per_line:
        uncovered = pd.concat(earliest_per_line, ignore_index=True)
    else:
        uncovered = pd.DataFrame()  # Empty DataFrame if no trips
    
    if uncovered.empty:
        result = None
    else:
        # Randomly select one trip from all the earliest trips
        result = uncovered.sample(n=1).iloc[0]
    
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
    ti: Optional[pd.Series] = None,
    tmax = 15.0,
    tmax_min = 0.0,
    inverse=False
) -> str:
    if (last_trip_id is None) == (ti is None):
        raise ValueError("Exactly one of last_trip_id or trip_object must be provided.")
    
    if last_trip_id is not None:
        ti = timetables[timetables.trip_id == last_trip_id].iloc[0]    
    
    def find_tmax(line):
        df = timetables[(timetables.line == line) & (timetables.covered == False)]['departure_interval']
        return df.max()
    
    
    ti_line = ti.line    
    #tmax = 20.0 #timetables[(timetables.line == ti.line) & (timetables.covered == False)]["departure_interval"].max()
    # try:
    #     int(tmax)
    # except:
    #     tmax = 0
    tj_start_cp = ti.dest_cp_id
    tj_candidates = timetables[
        (
            (timetables.start_cp_id == tj_start_cp)
                & (timetables.covered == False)
                & (timetables.departure_time >= eti + timedelta(minutes=tmax_min))
                & (timetables.departure_time <= eti + timedelta(minutes=tmax))
        )
    ].copy()
    # print()
    # print(tj_candidates)
    if inverse:
        try:
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

            tj_pre_candidates['lower_bound'] = eti + tj_pre_candidates['dh_time'] + timedelta(minutes=tmax_min)

            tj_pre_candidates['tmax'] = [timedelta(minutes=tmax)] * len(tj_pre_candidates) #tj_pre_candidates.apply(
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
        except:
            tj = tj_candidates.sample(random_state=seed)
    
    else:

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

            tj_pre_candidates['lower_bound'] = eti + tj_pre_candidates['dh_time'] + timedelta(minutes=tmax_min)

            tj_pre_candidates['tmax'] = [timedelta(minutes=tmax)] * len(tj_pre_candidates) #tj_pre_candidates.apply(
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

def random_tmax_select_next_trip(
    eti,
    timetables: pd.DataFrame,
    dh_times_df: pd.DataFrame,
    seed: Optional[int] = None,
    last_trip_id: Optional[str] = None,
    ti: Optional[pd.Series] = None,
    tmax = 15.0,
    tmax_min = 0.0,
    inverse = False
) -> str:
    if (last_trip_id is None) == (ti is None):
        raise ValueError("Exactly one of last_trip_id or trip_object must be provided.")
    
    if last_trip_id is not None:
        ti = timetables[timetables.trip_id == last_trip_id].iloc[0]    
    
    def find_tmax(line):
        df = timetables[(timetables.line == line) & (timetables.covered == False)]['departure_interval']
        return df.max()
    
    
    ti_line = ti.line    
    #tmax = 20.0 #timetables[(timetables.line == ti.line) & (timetables.covered == False)]["departure_interval"].max()
    # try:
    #     int(tmax)
    # except:
    #     tmax = 0
    tmax = float(np.random.randint(6,960))
    tj_start_cp = ti.dest_cp_id
    tj_candidates = timetables[
        (
            (timetables.start_cp_id == tj_start_cp)
                & (timetables.covered == False)
                & (timetables.departure_time >= eti + timedelta(minutes=tmax_min))
                & (timetables.departure_time <= eti + timedelta(minutes=tmax))
        )
    ].copy()
    # print()
    # print(tj_candidates)
    
    if inverse:
        try:
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

            tj_pre_candidates['lower_bound'] = eti + tj_pre_candidates['dh_time'] + timedelta(tmax_min)

            tj_pre_candidates['tmax'] = [timedelta(minutes=tmax)] * len(tj_pre_candidates) #tj_pre_candidates.apply(
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
        except:
            tj = tj_candidates.sample(random_state=seed)
    else:
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

            tj_pre_candidates['lower_bound'] = eti + tj_pre_candidates['dh_time'] + timedelta(tmax_min)

            tj_pre_candidates['tmax'] = [timedelta(minutes=tmax)] * len(tj_pre_candidates) #tj_pre_candidates.apply(
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

# Add this new function to your utils.py file


def post_process_timetable(timetables_df, lines_info):
    """
    Post-processes a generated timetable to standardize speeds and travel times.
    """
    print("\nApplying post-processing transformations to the timetable...")
    
    # Work with a copy
    df = timetables_df.copy()
    
    # Ensure 'departure_time' is a datetime object
    if not pd.api.types.is_datetime64_any_dtype(df['departure_time']):
        df['departure_time'] = pd.to_datetime(df['departure_time'])
    
    # Extract hour
    df['hour'] = df['departure_time'].dt.hour
    
    # STEP 1: Calculate hourly average speeds
    print("\nOriginal speeds sample (first 10):")
    print(df[['trip_id', 'hour', 'speed']].head(10))
    
    # Calculate average speed for each hour
    hourly_avg_speeds = df.groupby('hour')['speed'].mean().to_dict()
    print(f"\nHourly average speeds calculated: {hourly_avg_speeds}")
    
    # Replace each trip's speed with its hour's average
    df['original_speed'] = df['speed'].copy()  # Keep original for debugging
    df['speed'] = df['hour'].map(hourly_avg_speeds)
    
    print("\nAfter speed transformation (first 10):")
    print(df[['trip_id', 'hour', 'original_speed', 'speed']].head(10))
    
    # STEP 2: Recalculate travel times based on new speeds
    print("\nOriginal travel times sample (first 10):")
    print(df[['trip_id', 'line', 'planned_travel_time']].head(10))
    
    # Calculate new travel times
    new_travel_times = []
    for idx, row in df.iterrows():
        line = str(row['line'])
        distance = lines_info[line]['cp_distance_km']
        speed = row['speed']  # This is now the hourly average
        new_time = float(round((distance / speed) * 60))
        new_travel_times.append(new_time)
    
    df['original_travel_time'] = df['planned_travel_time'].copy()  # Keep original for debugging
    df['planned_travel_time'] = new_travel_times
    
    print("\nAfter travel time transformation (first 10):")
    print(df[['trip_id', 'line', 'original_travel_time', 'planned_travel_time']].head(10))
    
    # STEP 3: Clean up - remove temporary columns
    columns_to_drop = ['hour', 'original_speed', 'original_travel_time']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Verify the transformation worked
    print("\n=== VERIFICATION ===")
    # Check a sample hour to ensure all speeds are the same
    sample_hour_6 = df[df['departure_time'].dt.hour == 6][['trip_id', 'speed', 'planned_travel_time']].head(5)
    if not sample_hour_6.empty:
        print(f"Hour 6 trips (should all have same speed):")
        print(sample_hour_6)
        unique_speeds_h6 = sample_hour_6['speed'].nunique()
        print(f"Unique speeds in hour 6: {unique_speeds_h6} (should be 1)")
    
    print("\nPost-processing complete.\n")
    
    return df