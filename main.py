from config import *
from framework_v5 import main

if __name__ == "__main__":
    
    final_columns, final_objective = main(
        tmax_values_per_set=TMAX_VALUES_PER_SET,  # Use the correctly defined variable
        base_instance_path=None,  # None means generate new instances
        num_experiment_sets=NUM_EXPERIMENT_SETS,  # 30 sets
        k_values_per_set=K_VALUES_PER_SET,  # [50, 50, 50] for each set
        filter_graph=FILTER_GRAPH,  # False
        z_min=Z_MIN,  # 50
        max_iter=MAX_ITER  # 9
    )
    
    print("Program finished successfully!")
    print(f"Check 'experiments/' directory for all results and summaries.")
