#!/usr/bin/env python3
"""
Test script for the improved timetable generator for MD-EVSP
Generates timetables matching the Qingdao bus lines case study
"""

from datetime import datetime, timedelta
from utils import post_process_timetable
import pandas as pd
import numpy as np
from scipy import stats
import time
import os

# Copy the generate_timetable_v3 function here (or import it)
from utils import generate_timetable_v2  # Adjust import as needed
from inputs import *

def analyze_timetable(df, line_num):
    """Analyze generated timetable and compare with paper's patterns"""
    print(f"\n{'='*60}")
    print(f"ANALYSIS FOR LINE {line_num}")
    print(f"{'='*60}")
    
    # Basic statistics
    print(f"\nTotal trips: {len(df)}")
    print(f"First departure: {df.iloc[0]['departure_time_str']}")
    print(f"Last departure: {df.iloc[-1]['departure_time_str']}")
    
    # Analyze by CP
    for cp in df['start_cp'].unique():
        cp_df = df[df['start_cp'] == cp]
        print(f"\n{cp.upper()}: {len(cp_df)} trips")
        
    # Interval analysis
    intervals = df[df['departure_interval'] > 0]['departure_interval']
    print(f"\nDeparture Intervals:")
    print(f"  Mean: {intervals.mean():.1f} min")
    print(f"  Median: {intervals.median():.1f} min")
    print(f"  Min: {intervals.min():.1f} min")
    print(f"  Max: {intervals.max():.1f} min")
    print(f"  Std: {intervals.std():.1f} min")
    
    # Travel time analysis
    travel_times = df['planned_travel_time']
    print(f"\nPlanned Travel Times:")
    print(f"  Mean: {travel_times.mean():.1f} min")
    print(f"  Min: {travel_times.min():.1f} min")
    print(f"  Max: {travel_times.max():.1f} min")
    
    # Hour-by-hour breakdown
    print(f"\nHourly Distribution:")
    df['hour'] = pd.to_datetime(df['departure_time']).dt.hour
    hourly = df.groupby('hour').size()
    for hour, count in hourly.items():
        print(f"  {hour:02d}:00-{hour:02d}:59: {count:3d} trips")
    
    # Compare with Table 3 patterns (for line 4)
    if line_num == "4":
        print(f"\n--- Comparison with Table 3 (Line 4) ---")
        
        # Check early morning intervals (should be 7-8 minutes)
        early_trips = df[(pd.to_datetime(df['departure_time']).dt.hour < 7)]
        if len(early_trips) > 1:
            early_intervals = early_trips[early_trips['departure_interval'] > 0]['departure_interval']
            print(f"Early morning (before 7:00) intervals: {early_intervals.mean():.1f} min (Target: 7-8)")
        
        # Check travel times (should be ~70 normally, ~73 late evening)
        normal_trips = df[(pd.to_datetime(df['departure_time']).dt.hour >= 6) & 
                         (pd.to_datetime(df['departure_time']).dt.hour < 22)]
        late_trips = df[pd.to_datetime(df['departure_time']).dt.hour >= 22]
        
        if len(normal_trips) > 0:
            print(f"Normal hours travel time: {normal_trips['planned_travel_time'].mean():.1f} min (Target: 70)")
        if len(late_trips) > 0:
            print(f"Late evening travel time: {late_trips['planned_travel_time'].mean():.1f} min (Target: 73)")
    
    return df

def save_results(timetables, timestamp):
    """Save generated timetables to CSV files"""
    output_dir = "generated_timetables"
    os.makedirs(output_dir, exist_ok=True)
    
    # Combined timetable
    combined = pd.concat(timetables, ignore_index=True)
    combined_filename = f"{output_dir}/combined_timetable_{timestamp}.csv"
    combined.to_csv(combined_filename, index=False)
    print(f"\nCombined timetable saved to: {combined_filename}")
    
    # Individual timetables
    for i, (line_num, df) in enumerate(zip(["4", "59", "60"], timetables)):
        filename = f"{output_dir}/line_{line_num}_timetable_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"Line {line_num} timetable saved to: {filename}")
    
    return combined_filename

def main():
    """Main test function"""
    print("="*60)
    print("TIMETABLE GENERATOR TEST - MD-EVSP Qingdao Case Study")
    print("="*60)
    
    # Configuration for each line
    trips_configs = [
        {"line": "4", "total_trips": 290, "first_start_time": "05:50", "last_start_time": "22:35"},
        {"line": "59", "total_trips": 100, "first_start_time": "06:00", "last_start_time": "22:00"},
        {"line": "60", "total_trips": 120, "first_start_time": "06:00", "last_start_time": "22:00"}
    ]
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Generate timetables
    print("\nGenerating timetables...")
    start_time = time.time()
    
    timetables = []
    for config in trips_configs:
        print(f"\n--- Generating Line {config['line']} ---")
        df = generate_timetable_v2(lines_info, **config)
        df = post_process_timetable(df, lines_info)
        timetables.append(df)
        
    end_time = time.time()
    print(f"\nGeneration completed in {end_time - start_time:.2f} seconds")
    
    # Analyze each timetable
    for i, (config, df) in enumerate(zip(trips_configs, timetables)):
        analyze_timetable(df, config['line'])
    
    # Save results
    save_results(timetables, timestamp)
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    total_trips = sum(len(df) for df in timetables)
    print(f"Total trips generated: {total_trips}")
    print(f"Expected total: {sum(config['total_trips'] for config in trips_configs)}")
    
    # Check if we match the paper's trip counts
    for config, df in zip(trips_configs, timetables):
        line = config['line']
        expected = config['total_trips']
        actual = len(df)
        status = "✓" if actual == expected else "✗"
        print(f"Line {line}: {actual}/{expected} trips {status}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()