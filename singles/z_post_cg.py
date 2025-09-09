import pickle
import os
import datetime
import pandas as pd
from genetic_algorithm.genetic_algorithm_static_v2 import run_ga
import glob

# Default GA parameters for testing
DEFAULT_POST_CG_GA_PARAMS = {
    "max_gen": 20000,
    "crossover_p": 0.8,
    "mutation_p": 0.0001,
    "pop_size": 100,
    "elite_indivs": 5,
    "w1": 0.6,
    "w2": 0.6,
    "w3": 0.4,
    "w4": 1000,
    "sigma1": 0.01,
    "sigma2": 0.01,
    "sigma3": 0.02,
    "sigma4": 0.0001,
    "phi": 0.1,
    "phi_v": 0.1,
    "phi_r": 0.01,
    "phi_u": 0.001,
    "phi_ru": 0.01,
    "phi_s": 0.001,
    "gene_segments": 20,
    "L_min_constant": 140,
    "L_max_constant": 180,
    "use_objective": False,
}


def load_columns_from_file(file_path):
    """
    Load columns from a pickle file.
    
    Args:
        file_path: Path to the pickle file
    
    Returns:
        Dictionary containing columns and metadata
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def list_available_column_files():
    """
    List all available column files in the saved_columns directory.
    
    Returns:
        List of file paths
    """
    if not os.path.exists("saved_columns"):
        print("No 'saved_columns' directory found. Run the column generation orchestrator first.")
        return []
    
    files = glob.glob("saved_columns/*.pkl")
    return sorted(files)


def select_column_file():
    """
    Interactive function to select a column file.
    
    Returns:
        Selected file path or None
    """
    files = list_available_column_files()
    
    if not files:
        print("No column files found.")
        return None
    
    print("\nAvailable column files:")
    print("-" * 80)
    for i, file in enumerate(files, 1):
        filename = os.path.basename(file)
        print(f"{i}. {filename}")
    
    print("-" * 80)
    
    while True:
        try:
            choice = input(f"\nSelect file (1-{len(files)}) or 'q' to quit: ")
            if choice.lower() == 'q':
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                return files[idx]
            else:
                print(f"Please enter a number between 1 and {len(files)}")
        except ValueError:
            print("Invalid input. Please enter a number or 'q'")


def run_post_cg_ga_test(columns_file_path=None, ga_params=None, verbose=True):
    """
    Run post-CG GA on saved columns.
    
    Args:
        columns_file_path: Path to the columns file (if None, will prompt user)
        ga_params: GA parameters dictionary (if None, uses defaults)
        verbose: Whether to print progress
    
    Returns:
        Tuple of (best_fitness, best_cost, selected_columns)
    """
    
    # Select or load column file
    if columns_file_path is None:
        columns_file_path = select_column_file()
        if columns_file_path is None:
            print("No file selected. Exiting.")
            return None, None, None
    
    if not os.path.exists(columns_file_path):
        print(f"File not found: {columns_file_path}")
        return None, None, None
    
    # Load columns
    print("\n" + "=" * 80)
    print("LOADING COLUMNS")
    print("=" * 80)
    print(f"Loading from: {columns_file_path}")
    
    columns_data = load_columns_from_file(columns_file_path)
    columns = columns_data['columns']
    metadata = columns_data['metadata']
    
    print(f"✓ Loaded {len(columns)} columns")
    print(f"✓ Original timetables: {metadata['timetables_path']}")
    print(f"✓ Column generation optimal value: {metadata.get('optimal_value', 'N/A')}")
    print()
    
    # Get timetables path
    timetables_path = metadata['timetables_path']
    if not os.path.exists(timetables_path):
        print(f"Warning: Timetables file not found at {timetables_path}")
        timetables_path = input("Enter timetables path: ").strip()
        if not os.path.exists(timetables_path):
            print("Timetables file not found. Exiting.")
            return None, None, None
    
    # Use provided parameters or defaults
    if ga_params is None:
        ga_params = DEFAULT_POST_CG_GA_PARAMS.copy()
    
    # Run GA
    print("=" * 80)
    print("RUNNING POST-CG GA")
    print("=" * 80)
    print("GA Parameters:")
    for key, value in ga_params.items():
        print(f"  - {key}: {value}")
    print()
    
    # Create a dummy log dataframe (not saved to file)
    log_df = pd.DataFrame(columns=["experiment_name", "step", "metric", "value"])
    
    experiment_name = f"post_cg_test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        best_fitness, best_cost, selected_columns, _ = run_ga(
            columns,
            timetables_path,
            experiment_name,
            log_df,
            None,  # tmax_list
            ga_params,
            verbose=verbose
        )
        
        print()
        print("=" * 80)
        print("POST-CG GA COMPLETED")
        print("=" * 80)
        print(f"✓ Best fitness: {best_fitness:.4f}")
        print(f"✓ Best cost: {best_cost:.2f}")
        print(f"✓ Selected columns: {len(selected_columns)}")
        print(f"✓ Reduction: {len(columns)} → {len(selected_columns)} columns")
        print(f"✓ Reduction ratio: {len(selected_columns)/len(columns):.2%}")
        
        return best_fitness, best_cost, selected_columns
        
    except Exception as e:
        print(f"\nError running GA: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def run_parameter_sweep(columns_file_path=None):
    """
    Run multiple GA experiments with different parameters.
    
    Args:
        columns_file_path: Path to the columns file
    """
    
    # Define parameter sets to test
    parameter_sets = [
        {
            "name": "Default",
            "params": DEFAULT_POST_CG_GA_PARAMS.copy()
        },
        {
            "name": "High Mutation",
            "params": {
                **DEFAULT_POST_CG_GA_PARAMS,
                "mutation_p": 0.05,
            }
        },
        {
            "name": "Low Crossover",
            "params": {
                **DEFAULT_POST_CG_GA_PARAMS,
                "crossover_p": 0.5,
            }
        },
        {
            "name": "Large Population",
            "params": {
                **DEFAULT_POST_CG_GA_PARAMS,
                "pop_size": 200,
                "elite_indivs": 10,
            }
        },
        {
            "name": "Small Population Fast",
            "params": {
                **DEFAULT_POST_CG_GA_PARAMS,
                "pop_size": 50,
                "max_gen": 5000,
            }
        },
    ]
    
    # Select file once for all experiments
    if columns_file_path is None:
        columns_file_path = select_column_file()
        if columns_file_path is None:
            print("No file selected. Exiting.")
            return
    
    results = []
    
    print("\n" + "=" * 80)
    print("PARAMETER SWEEP")
    print("=" * 80)
    print(f"Testing {len(parameter_sets)} parameter configurations")
    print()
    
    for i, param_set in enumerate(parameter_sets, 1):
        print(f"\n[{i}/{len(parameter_sets)}] Testing: {param_set['name']}")
        print("-" * 40)
        
        best_fitness, best_cost, selected_columns = run_post_cg_ga_test(
            columns_file_path=columns_file_path,
            ga_params=param_set['params'],
            verbose=False  # Less verbose for sweep
        )
        
        if best_fitness is not None:
            results.append({
                'name': param_set['name'],
                'best_fitness': best_fitness,
                'best_cost': best_cost,
                'selected_columns': len(selected_columns) if selected_columns else 0
            })
    
    # Print summary
    print("\n" + "=" * 80)
    print("PARAMETER SWEEP RESULTS")
    print("=" * 80)
    
    if results:
        results_df = pd.DataFrame(results)
        print(results_df.to_string())
        
        # Find best
        best_idx = results_df['best_fitness'].idxmax()
        print(f"\nBest configuration: {results_df.loc[best_idx, 'name']}")
        print(f"  - Fitness: {results_df.loc[best_idx, 'best_fitness']:.4f}")
        print(f"  - Cost: {results_df.loc[best_idx, 'best_cost']:.2f}")
        print(f"  - Columns: {results_df.loc[best_idx, 'selected_columns']}")


def main():
    """
    Main function with interactive menu.
    """
    print("\n" + "=" * 80)
    print("POST-COLUMN GENERATION GA TESTER")
    print("=" * 80)
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    while True:
        print("\nOptions:")
        print("1. Run single GA test with default parameters")
        print("2. Run single GA test with custom parameters")
        print("3. Run parameter sweep")
        print("4. List available column files")
        print("q. Quit")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == '1':
            run_post_cg_ga_test()
            
        elif choice == '2':
            # Allow user to modify parameters
            params = DEFAULT_POST_CG_GA_PARAMS.copy()
            
            print("\nCurrent parameters:")
            for key, value in params.items():
                print(f"  {key}: {value}")
            
            print("\nEnter parameter changes (format: key=value, or press Enter to use defaults):")
            print("Example: pop_size=200")
            
            while True:
                change = input("Parameter change (or Enter to continue): ").strip()
                if not change:
                    break
                
                try:
                    key, value = change.split('=')
                    key = key.strip()
                    
                    if key in params:
                        # Try to convert to appropriate type
                        if isinstance(params[key], bool):
                            params[key] = value.lower() in ['true', '1', 'yes']
                        elif isinstance(params[key], int):
                            params[key] = int(value)
                        elif isinstance(params[key], float):
                            params[key] = float(value)
                        else:
                            params[key] = value
                        
                        print(f"  ✓ Set {key} = {params[key]}")
                    else:
                        print(f"  ⚠ Unknown parameter: {key}")
                        
                except ValueError:
                    print("  ⚠ Invalid format. Use: key=value")
            
            run_post_cg_ga_test(ga_params=params)
            
        elif choice == '3':
            run_parameter_sweep()
            
        elif choice == '4':
            files = list_available_column_files()
            if files:
                print("\nAvailable column files:")
                for file in files:
                    print(f"  - {os.path.basename(file)}")
            else:
                print("No column files found.")
                
        elif choice.lower() == 'q':
            print("\nExiting...")
            break
            
        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    main()