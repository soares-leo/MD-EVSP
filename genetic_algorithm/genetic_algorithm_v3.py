from experiments.exp_set_01_2025_08_07_01_42_59.columns_data import columns
import pandas as pd
import numpy as np
import math

# --- GA Parameters ---
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

# --- Pre-computation for Efficiency ---
timetable_trips = pd.read_csv("initializer/files/instance_2025-08-07_01-42-34_l4_290_l59_100_l60_120.csv", usecols = ["trip_id"])

## OPTIMIZATION: Create a set of all required trips ONCE outside the main loop.
## This avoids repeatedly iterating over the timetable_trips DataFrame.
REQUIRED_TRIPS_SET = set(timetable_trips["trip_id"].values)

## OPTIMIZATION: Use a dictionary as a cache (memoization) for the fitness function.
## If the same individual is evaluated twice, we can return the stored result instantly.
FITNESS_CACHE = {}

def initialize_population():
    initial_population = []
    print(f"Initializing population of size {pop_size}...")
    while len(initial_population) < pop_size:
        ## OPTIMIZATION: Generate the individual using a single, vectorized NumPy operation
        ## instead of a Python loop or list comprehension.
        individual = (np.random.uniform(0.0, 1.0, len(columns)) >= 0.7).astype(np.int8)
        
        vehicles = individual.sum() # Use NumPy's fast sum
        if vehicles <= L_max and vehicles >= L_min:
            initial_population.append(individual)
    return initial_population

def evaluate_individial(individual):
    ## OPTIMIZATION: Check the cache first. If we've seen this individual before,
    ## return the cached fitness and cost to avoid re-calculating.
    # Individuals are NumPy arrays, which are not hashable. Convert to a tuple to use as a key.
    individual_tuple = tuple(individual)
    if individual_tuple in FITNESS_CACHE:
        return FITNESS_CACHE[individual_tuple]

    v = individual.sum()
    cost = 0
    operation_times = []
    flat_trips = []
    
    # This loop depends on the complex 'columns' structure, so vectorization is difficult here.
    # We will focus on optimizing the parts after the loop.
    for i, gene in enumerate(individual):
        if gene == 1:
            k = col_keys[i]
            flat_trips.extend(columns[k]['Path'])
            cost += columns[k]["Cost"]
            data = columns[k]["Data"]
            op_time = sum([float(data["total_dh_time"]), float(data["total_travel_time"]), float(data["total_wait_time"])])
            operation_times.append(float(op_time))

    flat_trips = [item for item in flat_trips if item.startswith("l")]
    unique_trips = set(flat_trips)
    
    r = len(flat_trips) - len(unique_trips)
    
    ## OPTIMIZATION: This is the highest-impact change.
    ## Calculate uncovered trips (u) using a highly optimized set difference operation
    ## instead of a slow filter/lambda function over the entire timetable.
    u = len(REQUIRED_TRIPS_SET - unique_trips)
    
    s = sum(1 for item in operation_times if item < 480) # Using a generator expression is slightly more memory efficient
    
    # Avoid division by zero if the denominator happens to be 0
    denominator = (w1 * v) + (w2 * (r + u)) + (w3 * s)
    if denominator == 0:
        fitness = math.inf
    else:
        fitness = w4 / denominator

    ## OPTIMIZATION: Store the result in the cache before returning.
    FITNESS_CACHE[individual_tuple] = (fitness, cost)
    
    return fitness, cost

def spin_roulette(fitness_list, population):
    ## OPTIMIZATION: Convert fitness_list to a NumPy array for fast vectorized calculations.
    fitness_array = np.array(fitness_list)
    total_fitness = fitness_array.sum()
    
    if total_fitness == 0:
        # If all fitness is zero, choose uniformly.
        probs = None
    else:
        # Calculate probabilities with a single vectorized operation.
        probs = fitness_array / total_fitness
        
    parent_indices = np.random.choice(a=len(population), size=2, p=probs, replace=False)
    # Return NumPy arrays directly
    return population[parent_indices[0]], population[parent_indices[1]]

def make_crossover(parents, genes_per_segment):
    parent_1, parent_2 = parents
    offspring_1 = []
    offspring_2 = []
    
    num_genes = len(parent_1)
    # This loop is small (not the main bottleneck), so we leave its logic for clarity.
    for i in range(0, num_genes, genes_per_segment):
        p1_segment = parent_1[i:i + genes_per_segment]
        p2_segment = parent_2[i:i + genes_per_segment]
        
        if np.random.uniform(0.0, 1.0) <= 0.5:
            offspring_1.extend(p2_segment)
            offspring_2.extend(p1_segment)
        else:
            offspring_1.extend(p1_segment)
            offspring_2.extend(p2_segment)
            
    return np.array(offspring_1, dtype=np.int8), np.array(offspring_2, dtype=np.int8)

def mutate(individual):
    ## OPTIMIZATION: Use NumPy boolean masking for vectorized mutation.
    ## This is significantly faster than iterating gene by gene.
    mutation_mask = np.random.uniform(0.0, 1.0, len(individual)) <= mutation_p
    # Use XOR (^) to flip the bits (0->1, 1->0) only where the mask is True.
    individual[mutation_mask] = 1 - individual[mutation_mask]
    return individual

# The main run_ga function uses the optimized helper functions above.
# The algorithmic logic remains identical to the corrected version.
def run_ga():
    g = 1
    best_fit = -math.inf
    best_cost = math.inf
    best_individual_so_far = None # Initialize as None
    
    elite_population = []
    elite_fitness = []
    elite_costs = []
    
    while g <= max_gen:
        print()
        print(f"GENERATION {g} --------------------------------------------------")
        
        if g == 1:
            population = initialize_population()
            fitness_list = []
            cost_list = []
            for ind in population:
                eval_result = evaluate_individial(ind)
                fitness_list.append(eval_result[0])
                cost_list.append(eval_result[1])
            
            # Convert to NumPy array for faster operations later
            population = np.array(population, dtype=np.int8)

            elite_indices = np.argsort(fitness_list)[-elite_indivs:]
            elite_population = [population[i] for i in elite_indices]
            elite_fitness = [fitness_list[i] for i in elite_indices]
            elite_costs = [cost_list[i] for i in elite_indices]

            best_idx = elite_indices[-1]
            best_fit = fitness_list[best_idx]
            best_cost = cost_list[best_idx]
            best_individual_so_far = population[best_idx]

            print(f"Best fit: {best_fit:.4f} | Best cost: {best_cost:.2f}")
            g += 1
            continue

        print(f"Best fit so far: {best_fit:.4f} | Best cost: {best_cost:.2f} | Cache size: {len(FITNESS_CACHE)}")
        
        previous_best_idx = np.argmax(fitness_list)
        previous_best_individual = population[previous_best_idx]
        previous_best_fitness = fitness_list[previous_best_idx]

        new_population = np.empty_like(population)
        for i in range(0, pop_size, 2):
            parent_1, parent_2 = spin_roulette(fitness_list, population)
            
            if np.random.uniform(0.0, 1.0) <= crossover_p:
                offspring_1, offspring_2 = make_crossover((parent_1, parent_2), gene_segments)
            else:
                offspring_1, offspring_2 = parent_1.copy(), parent_2.copy()

            new_population[i] = mutate(offspring_1)
            if i + 1 < pop_size:
                new_population[i+1] = mutate(offspring_2)

        new_fitness_list = []
        new_cost_list = []
        for ind in new_population:
            eval_result = evaluate_individial(ind)
            new_fitness_list.append(eval_result[0])
            new_cost_list.append(eval_result[1])

        new_elite_indices = np.argsort(new_fitness_list)[-elite_indivs:]
        new_elite_population = [new_population[i] for i in new_elite_indices]
        new_elite_fitness = [new_fitness_list[i] for i in new_elite_indices]
        new_elite_costs = [new_cost_list[i] for i in new_elite_indices]

        temp_elite_pop = elite_population + new_elite_population
        temp_elite_fit = elite_fitness + new_elite_fitness
        temp_elite_cost = elite_costs + new_elite_costs
        
        master_elite_indices = np.argsort(temp_elite_fit)[-elite_indivs:]
        elite_population = [temp_elite_pop[i] for i in master_elite_indices]
        elite_fitness = [temp_elite_fit[i] for i in master_elite_indices]
        elite_costs = [temp_elite_cost[i] for i in master_elite_indices]
        
        new_best_fitness = max(new_fitness_list)
        worst_new_idx = np.argmin(new_fitness_list)
        
        if new_best_fitness > previous_best_fitness:
            new_population[worst_new_idx] = previous_best_individual
            new_fitness_list[worst_new_idx] = previous_best_fitness
            new_cost_list[worst_new_idx] = cost_list[previous_best_idx]
        else:
            total_elite_fitness = sum(elite_fitness)
            if total_elite_fitness > 0:
                elite_probs = [f / total_elite_fitness for f in elite_fitness]
                chosen_elite_master_idx = np.random.choice(len(elite_population), p=elite_probs)
                
                new_population[worst_new_idx] = elite_population[chosen_elite_master_idx]
                new_fitness_list[worst_new_idx] = elite_fitness[chosen_elite_master_idx]
                new_cost_list[worst_new_idx] = elite_costs[chosen_elite_master_idx]

        population = new_population
        fitness_list = new_fitness_list
        cost_list = new_cost_list
        
        current_best_idx = np.argmax(fitness_list)
        if fitness_list[current_best_idx] > best_fit:
            best_fit = fitness_list[current_best_idx]
            best_cost = cost_list[current_best_idx]
            best_individual_so_far = population[current_best_idx]
        
        g += 1

# Example of how to run it
if __name__ == '__main__':
    run_ga()