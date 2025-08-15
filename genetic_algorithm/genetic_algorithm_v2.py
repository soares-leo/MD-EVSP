from experiments.exp_set_01_2025_08_07_01_42_59.columns_data import columns
import pandas as pd
import numpy as np
import math

L_min = 140
L_max = 180
max_gen = 20000
crossover_p = 0.3
mutation_p = 0.01
pop_size = 100
elite_indivs = 5
w1, w2, w3, w4 = 0.6, 0.6, 0.4, 1000
gens = 1
census = 0
E_num = 5
gene_segments = 20
col_keys = list(columns.keys())

timetable_trips = pd.read_csv("initializer/files/instance_2025-08-07_01-42-34_l4_290_l59_100_l60_120.csv", usecols = ["trip_id"])

def initialize_population():
    initial_population = []
    print(len(columns))
    while len(initial_population) < pop_size:
        #print(len(initial_population))
        individual = np.random.uniform(0.0, 1.0, len(columns))
        individual = [1 if x >= 0.7 else 0 for x in individual]
        vehicles = sum(individual)
        if vehicles <= L_max and vehicles >= L_min:
            initial_population.append(individual)
        else:
            continue
    return initial_population

def evaluate_individial(individual):
    v = sum(individual)
    cost = 0
    operation_times = []
    flat_trips = []
    for i, gene in enumerate(individual):
        if gene == 0:
            continue
        else:
            k = col_keys[i]
            flat_trips.extend(columns[k]['Path'])
            cost += columns[k]["Cost"]
            data = columns[k]["Data"]
            op_time = sum([float(data["total_dh_time"]), float(data["total_travel_time"]), float(data["total_wait_time"])])
            operation_times.append(float(op_time))
    flat_trips = [item for item in flat_trips if item.startswith("l")]

    
    flat_trips_len = len(flat_trips)
    unique_trips = set(flat_trips)
    #print("v:", v)
    # print(len(flat_trips))
    # print(len(unique_trips))
    r = flat_trips_len - len(unique_trips)
    u = len(list(filter(lambda x: x not in unique_trips, timetable_trips["trip_id"].values)))
    s = sum([1 if item < 480 else 0 for item in operation_times])
    fitness = w4 / ((w1 * v) + (w2 * (r + u)) + (w3 * s))
    #print("v:", v, "r:", r, "u:", u, "s:", s)
    # print(f"f(x) = {w4} / (({w1} * {v}) + ({w2} * ({r} + {u})) + ({w3} * {s}))")
    return fitness, cost

def spin_roulette(fitness_list, population):
    total_fitness = sum(fitness_list)
    probs = [x / total_fitness for x in fitness_list]
    parents_indexes = np.random.choice(a=len(probs), size=2, p=probs, replace=False)
    parents = [population[x] for x in parents_indexes]
    return parents

def make_crossover(parents, genes_per_segment):
    parent_1 = parents[0]
    parent_2 = parents[1]
    segmented_parent_1 = []
    segmented_parent_2 = []
    left_bound = 0
    right_bound = genes_per_segment
    segments = math.ceil(len(parent_1) / genes_per_segment)
    for i in range(segments):
        parent_1_segment = parent_1[left_bound:right_bound]
        parent_2_segment = parent_2[left_bound:right_bound]
        segmented_parent_1.append(parent_1_segment)
        segmented_parent_2.append(parent_2_segment)
        left_bound += genes_per_segment
        right_bound += genes_per_segment

    offspring_1 = []
    offspring_2 = []
    
    for p1_segment, p2_segment in zip(segmented_parent_1, segmented_parent_2):
        u = np.random.uniform(0.0, 1.0)
        if u <= 0.5:
            offspring_1.extend(p2_segment)
            offspring_2.extend(p1_segment)
        else:
            offspring_1.extend(p1_segment)
            offspring_2.extend(p2_segment)
            
    return offspring_1, offspring_2

def mutate(individual):
    for i, gene in enumerate(individual):
        u = np.random.uniform(0.0, 1.0)
        if u <= mutation_p:
            if gene == 0:
                individual[i] = 1
            else:
                individual[i] = 0
    return individual

    #sorted_fitness = np.argsort(fitness_list)

def run_ga():
    g = 1
    # Global best trackers
    best_fit = -math.inf
    best_cost = math.inf
    best_individual_so_far = []
    
    # Initialize master elite pool (will be populated in Generation 1)
    elite_population = []
    elite_fitness = []
    elite_costs = []
    
    # --- Main Evolutionary Loop ---
    while g <= max_gen:
        print()
        print(f"GENERATION {g} --------------------------------------------------")
        
        # --- Generation 1: Initialization ---
        if g == 1:
            population = initialize_population()
            fitness_list = []
            cost_list = []
            for ind in population:
                eval_result = evaluate_individial(ind)
                fitness_list.append(eval_result[0])
                cost_list.append(eval_result[1])
            
            # Initialize the master elite pool with the best from the first generation
            elite_indices = np.argsort(fitness_list)[-elite_indivs:]
            elite_population = [population[i] for i in elite_indices]
            elite_fitness = [fitness_list[i] for i in elite_indices]
            elite_costs = [cost_list[i] for i in elite_indices]

            # Set the initial "best so far"
            best_idx = elite_indices[-1]
            best_fit = fitness_list[best_idx]
            best_cost = cost_list[best_idx]
            best_individual_so_far = population[best_idx]

            print(f"Best fit: {best_fit} | Best cost: {best_cost}")
            g += 1
            continue

        print(f"Best fit so far: {best_fit} | Best cost: {best_cost}")
        
        # --- Generations > 1: Evolution ---

        # 1. Store the previous generation's best solution details
        previous_best_idx = np.argmax(fitness_list)
        previous_best_individual = population[previous_best_idx]
        previous_best_fitness = fitness_list[previous_best_idx]

        # 2. Create the new population via selection, crossover, and mutation
        new_population = []
        while len(new_population) < pop_size:
            parents = spin_roulette(fitness_list, population)
            
            # Perform crossover with probability crossover_p
            u_cross = np.random.uniform(0.0, 1.0)
            if u_cross <= crossover_p:
                offspring_1, offspring_2 = make_crossover(parents, gene_segments)
            else:
                offspring_1, offspring_2 = parents[0][:], parents[1][:] # Use copies

            # Mutate the offspring
            offspring_1 = mutate(offspring_1)
            offspring_2 = mutate(offspring_2)
            
            new_population.append(offspring_1)
            if len(new_population) < pop_size:
                new_population.append(offspring_2)

        # 3. Evaluate the new population
        new_fitness_list = []
        new_cost_list = []
        for ind in new_population:
            eval_result = evaluate_individial(ind)
            new_fitness_list.append(eval_result[0])
            new_cost_list.append(eval_result[1])

        # 4. Implement the Elitist Strategy from the paper
        
        # 4.1. Update the Master Elite List
        # Get the top E individuals from the newly created population
        new_elite_indices = np.argsort(new_fitness_list)[-elite_indivs:]
        new_elite_population = [new_population[i] for i in new_elite_indices]
        new_elite_fitness = [new_fitness_list[i] for i in new_elite_indices]
        new_elite_costs = [new_cost_list[i] for i in new_elite_indices]

        # [cite_start]Combine old master elites with new elites to form a temporary pool [cite: 483]
        temp_elite_pop = elite_population + new_elite_population
        temp_elite_fit = elite_fitness + new_elite_fitness
        temp_elite_cost = elite_costs + new_elite_costs
        
        # [cite_start]Sort the temporary pool and select the top E to be the new master elite list [cite: 483]
        master_elite_indices = np.argsort(temp_elite_fit)[-elite_indivs:]
        elite_population = [temp_elite_pop[i] for i in master_elite_indices]
        elite_fitness = [temp_elite_fit[i] for i in master_elite_indices]
        elite_costs = [temp_elite_cost[i] for i in master_elite_indices]
        
        # 4.2. Perform Replacement in the new population
        new_best_fitness = max(new_fitness_list)
        worst_new_idx = np.argmin(new_fitness_list)
        
        if new_best_fitness > previous_best_fitness:
            # Case 1: New generation is better. [cite_start]Replace its worst with the PREVIOUS best[cite: 484].
            new_population[worst_new_idx] = previous_best_individual
            new_fitness_list[worst_new_idx] = previous_best_fitness
            new_cost_list[worst_new_idx] = cost_list[previous_best_idx]
        else:
            # Case 2: New generation is not better. Replace its worst with an individual
            # [cite_start]selected from the MASTER ELITE POOL using roulette wheel selection[cite: 484].
            total_elite_fitness = sum(elite_fitness)
            if total_elite_fitness > 0:
                elite_probs = [f / total_elite_fitness for f in elite_fitness]
                # Choose one individual from the master elite list based on its fitness
                chosen_elite_master_idx = np.random.choice(len(elite_population), p=elite_probs)
                
                # Replace the worst of the new population
                new_population[worst_new_idx] = elite_population[chosen_elite_master_idx]
                new_fitness_list[worst_new_idx] = elite_fitness[chosen_elite_master_idx]
                new_cost_list[worst_new_idx] = elite_costs[chosen_elite_master_idx]

        # 5. The modified new population becomes the population for the next generation
        population = new_population
        fitness_list = new_fitness_list
        cost_list = new_cost_list
        
        # 6. Update the overall best-so-far solution
        current_best_idx = np.argmax(fitness_list)
        if fitness_list[current_best_idx] > best_fit:
            best_fit = fitness_list[current_best_idx]
            best_cost = cost_list[current_best_idx]
            best_individual_so_far = population[current_best_idx]
        
        g += 1
    
run_ga()




