import random
import numpy as np


def run_ga(S, Lmin, Lmax, pop_size, max_gen, crossover_prob, mutation_prob, elite_size):
    """
    Genetic Algorithm to select a subset of columns (routes) from S.

    S: dict {route_key: {"Path":..., "Cost":..., "ReducedCost":..., "Data":{...}}}
    Returns: tuple (selected_routes_dict, optimal_fitness)
    """
    # print initial parameters
    print(f"Starting GA with parameters: original Lmin={Lmin}, original Lmax={Lmax}, pop_size={pop_size}, max_gen={max_gen}, crossover_prob={crossover_prob}, mutation_prob={mutation_prob}, elite_size={elite_size}")

    # --- Prepare data ---
    keys = list(S.keys())
    n = len(keys)
    print(f"Number of routes (genes): {n}")

    # ensure Lmin, Lmax feasible relative to n
    Lmin_eff = max(1, min(Lmin, n))
    Lmax_eff = min(Lmax, n)
    if Lmin_eff > Lmax_eff:
        Lmin_eff = Lmax_eff
    print(f"Adjusted vehicle bounds: Lmin={Lmin_eff}, Lmax={Lmax_eff}")

    # fitness weights (per paper) citeturn15file1
    w1, w2, w3, w4 = 0.6, 0.6, 0.4, 1000
    # mapping of short-block thresholds by bus line id
    short_map = {4: 4, 59: 6, 60: 6}

    # --- Solution coding: binary vector length n citeturn15file0
    def random_individual():
        while True:
            ind = [random.choice([0,1]) for _ in range(n)]
            c = sum(ind)
            if Lmin_eff <= c <= Lmax_eff:
                return ind

    # --- Fitness function citeturn15file1
    def fitness(ind):
        v = sum(ind)
        time_points = {}
        for gene, k in zip(ind, keys):
            if gene:
                for trip in S[k]["Path"]:
                    time_points[trip] = time_points.get(trip, 0) + 1
        r = sum(cnt-1 for cnt in time_points.values() if cnt>1)
        all_trips = set(trip for route in S.values() for trip in route["Path"])
        u = len(all_trips - set(time_points.keys()))
        s = 0
        for gene, k in zip(ind, keys):
            if gene:
                first_trip = S[k]["Path"][0]
                try:
                    line_id = int(first_trip[1:].split("_")[0])
                except:
                    line_id = None
                short_thresh = short_map.get(line_id, 5)
                if len(S[k]["Path"]) < short_thresh:
                    s += 1
        return w1/v + w2*r + w3*u + w4*s

    # --- Initialize population citeturn15file1
    population = [random_individual() for _ in range(pop_size)]
    print(f"Initialized population with {len(population)} individuals")
    prev_elite = []

    # --- GA main loop ---
    for gen in range(1, max_gen+1):
        # Evaluate fitness
        fits = [fitness(ind) for ind in population]
        best_fit = max(fits)
        avg_fit = np.mean(fits)
        print(f"Generation {gen:>5}: best fitness = {best_fit:.4f}, average fitness = {avg_fit:.4f}")

        # Selection: roulette wheel
        total_fit = sum(fits)
        probs = [f/total_fit for f in fits]
        selected = [population[np.random.choice(range(pop_size), p=probs)] for _ in range(pop_size)]

        # Crossover citeturn15file3
        next_pop = []
        for i in range(0, pop_size, 2):
            p1, p2 = selected[i], selected[min(i+1, pop_size-1)]
            if random.random() < crossover_prob:
                seg = 20
                c1, c2 = p1.copy(), p2.copy()
                for start in range(0, n, seg):
                    if random.random() < 0.5:
                        end = min(start+seg, n)
                        c1[start:end], c2[start:end] = c2[start:end], c1[start:end]
                next_pop += [c1, c2]
            else:
                next_pop += [p1, p2]

        # Mutation citeturn15file3
        for ind in next_pop:
            for j in range(n):
                if random.random() < mutation_prob:
                    ind[j] = 1 - ind[j]

        # Enforce Lmin_eff/Lmax_eff
        population = [ind if Lmin_eff<=sum(ind)<=Lmax_eff else random_individual() for ind in next_pop]

        # Elitist strategy citeturn15file3
        fits = [fitness(ind) for ind in population]
        elite_idxs = sorted(range(pop_size), key=lambda i: fits[i], reverse=True)[:elite_size]
        curr_elite = [population[i] for i in elite_idxs]
        merged = prev_elite + curr_elite
        merged_sorted = sorted(((ind, fitness(ind)) for ind in merged), key=lambda x: x[1], reverse=True)
        prev_elite = [ind for ind,_ in merged_sorted[:elite_size]]
        worst_idx = min(range(pop_size), key=lambda i: fits[i])
        population[worst_idx] = random.choice(prev_elite)

    # --- Final selection ---
    final_fits = [fitness(ind) for ind in population]
    best_idx = max(range(pop_size), key=lambda i: final_fits[i])
    best_ind = population[best_idx]
    optimal_fitness = final_fits[best_idx]
    print(f"GA completed: optimal fitness = {optimal_fitness:.4f}")

    # Build result dict
    result = {k: S[k] for gene, k in zip(best_ind, keys) if gene}
    return result, optimal_fitness
