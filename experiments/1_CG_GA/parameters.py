_1_CG_GA_PARAMS = {
    "solution_generator": {
        "timetables_path": "generated_timetables/combined_timetable_20250907_022354.csv",
        "tmax": 20,
        "seed": None,
        "random_start_trip": "no",
        "random_tmax": False,
        "tmax_min": 0,
        "inverse": False,
    },
    "CG": {
        "z_min": 50,
        "k": 100,
        "max_iter": 10,
        "filter_graph": True,
    },
    "GA": {
        "max_gen": 20000,
        "crossover_p": 0.3,
        "mutation_p": 0.01,
        "pop_size": 100,
        "elite_indivs": 5,
        "gene_segments": 20,
        "w1": 0.6,
        "w2": 0.6,
        "w3": 0.4,
        "w4": 1000,
        "L_min_constant": 140,
        "L_max_constant": 180,
        "use_objective": False
    }
}