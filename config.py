# Paths and Files
CPLEX_PATH = "C:/Program Files/IBM/ILOG/CPLEX_Studio2211/cplex/bin/x64_win64/cplex.exe"
DEFAULT_INSTANCE_PATH = None
ALTERNATIVE_INSTANCE_PATH = "initializer/files/instance_2025-08-16_00-10-31_l4_290_l59_100_l60_120.csv"

TRIPS_CONFIGS = [
    {
    "line": 4,
    "total_trips": 30,
    "first_start_time": "05:50",
    "last_start_time": "22:50",
    },
    {
    "line": 59,
    "total_trips": 10,
    "first_start_time": "06:40",
    "last_start_time": "19:50",
    },
    {
    "line": 60,
    "total_trips": 12,
    "first_start_time": "06:00",
    "last_start_time": "21:10",
    },
]

# Algorithm Parameters
DEADHEAD_SPEED = 20  # km/h for deadhead calculations
RANDOM_SEED = 1
MAX_DRIVING_TIME = 120  # D parameter in minutes
MAX_TOTAL_TIME = 16 * 60  # T_d parameter in minutes

# Experiment Sets Configuration
NUM_EXPERIMENT_SETS = 2  # Number of experiment sets to run
K_VALUES_PER_SET = [50, 50]  # K values to test in each set
TMAX_VALUES_PER_SET = [10.0, 30.0]

# Generate experiment set IDs and notes automatically
EXP_SET_IDS = [f"experiment_set_{i+1:03d}" for i in range(NUM_EXPERIMENT_SETS)]
EXP_NOTES = [f"Experiment set {i+1} with K values {K_VALUES_PER_SET}" for i in range(NUM_EXPERIMENT_SETS)]

# Experiment Parameters for each experiment within a set
EXPERIMENTS_PER_SET = len(K_VALUES_PER_SET)  # 3 experiments per set
FILTER_GRAPH = False  # Same for all experiments
Z_MIN = 50  # Convergence threshold
MAX_ITER = 9  # Maximum iterations