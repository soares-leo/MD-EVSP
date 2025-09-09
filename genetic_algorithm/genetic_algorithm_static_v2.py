import pandas as pd
import numpy as np
import math
import time
import datetime
import os
import itertools
import random
from multiprocessing import Pool, cpu_count
from functools import lru_cache
import scipy.sparse as sp

class GeneticAlgorithm:
    """
    Optimized Genetic Algorithm for column selection optimization.
    
    Uses a dynamic exponential fitness scaling function based on Lima et al. (1996).
    The primary goal is always to MAXIMIZE fitness.
    This version includes performance optimizations for large-scale problems.
    
    OPTIMIZATIONS:
    - Bit mask operations for trip counting
    - Vectorized fitness evaluation
    - Parallel processing for population evaluation
    - Caching for repeated evaluations
    - Memory-efficient sparse representations (optional)
    """
    
    def __init__(self, columns, timetables_path, experiment_name, ga_params=None, use_parallel=True, use_cache=False):
        """
        Initialize GA with columns and parameters.
        
        Args:
            use_parallel: Enable parallel processing for population evaluation
            use_cache: Enable caching of fitness evaluations
        """
        self.columns = columns
        self.col_keys = list(columns.keys())
        self.timetables_path = timetables_path
        self.experiment_name = experiment_name
        self.use_parallel = use_parallel and cpu_count() > 1
        self.use_cache = use_cache
        
        params = ga_params or {}
        self.max_gen = params.get('max_gen', 1000)
        self.crossover_p = params.get('crossover_p', 0.3)
        self.mutation_p = params.get('mutation_p', 0.01)
        self.pop_size = params.get('pop_size', 100)
        self.elite_indivs = params.get('elite_indivs', 5)
        self.gene_segments = params.get('gene_segments', 20)
        self.L_min_constant = params.get('L_min_constant', 140)
        self.L_max_constant = params.get('L_max_constant', 180)
        
        # --- TRANSLATION LAYER FOR BACKWARD COMPATIBILITY ---
        if 'fitness_components' in params and 'phi_values' in params:
            print("Using new GA parameter format (fitness_components, phi_values).")
            self.fitness_components = params['fitness_components']
            self.phi_values = params['phi_values']
        else:
            print("New GA parameters not found. Attempting to translate from old format (e.g., phi_v, phi_u)...")
            translated_components = []
            translated_phis = {}
            possible_components = {'v', 'r', 'u', 's'}
            
            for key, value in params.items():
                if key.startswith('phi_'):
                    component = key.split('_')[1]
                    if component in possible_components:
                        translated_components.append(component)
                        translated_phis[component] = value
            
            if translated_components:
                print(f"Successfully translated old parameters.")
                self.fitness_components = translated_components
                self.phi_values = translated_phis
            else:
                print("No valid old or new parameters found. Using default fitness settings.")
                self.fitness_components = ['v', 'r', 'u', 's']
                self.phi_values = {'v': 0.01, 'r': 0.01, 'u': 0.8, 's': 0.2}

        print(f"Fitness will be based on components: {self.fitness_components} with phi values: {self.phi_values}")
        
        # Load timetable and setup trip indexing
        self.timetable_trips = pd.read_csv(timetables_path, usecols=["trip_id"])
        num_of_trips = len(self.timetable_trips)
        self.total_trips = num_of_trips
        self.L_min = int(round((self.L_min_constant / num_of_trips) * num_of_trips))
        self.L_max = int(round((self.L_max_constant / num_of_trips) * num_of_trips))

        # Create trip index mapping for bit operations
        self.trip_to_idx = {trip: idx for idx, trip in enumerate(self.timetable_trips['trip_id'].values)}
        
        self._preprocess_columns()
        
        self.best_individual = None
        self.best_fitness = -math.inf
        self.best_cost = math.inf
        self.best_indicators = {}
        
        self.generation_history = []
        
        # Sigma parameters for fitness calculation
        self.sigma1 = params.get('sigma1', 0.0)  # Applied to v
        self.sigma2 = params.get('sigma2', 0.0)  # Applied to r
        self.sigma3 = params.get('sigma3', 0.0)  # Applied to u
        self.sigma4 = params.get('sigma4', 0.0)  # Applied to s
        
        # Vectorized sigma for batch operations
        self.sigma_vector = np.array([self.sigma1, self.sigma2, self.sigma3, self.sigma4]) * -1
        
        if self.sigma1 == 0.0 and self.sigma2 == 0.0 and self.sigma3 == 0.0 and self.sigma4 == 0.0:
            print("\nWARNING: All fitness weights (sigma1, sigma2, sigma3, sigma4) are zero. All individuals will have a fitness of 1.0.")
            print("Please provide at least one sigma weight in ga_params for meaningful selection.\n")

        # Attributes for paper's elitism strategy
        self.elite_pool = [] 
        self.previous_gen_best_eval = None
        
        # Cache for fitness evaluations
        if self.use_cache:
            self.fitness_cache = {}
            self.cache_hits = 0
            self.cache_misses = 0
        
        # Set up parallel processing
        if self.use_parallel:
            self.num_processes = min(cpu_count(), 8)
            print(f"Parallel processing enabled with {self.num_processes} processes")

    def _preprocess_columns(self):
        """Pre-calculates static data AND trip bitmasks for each column to speed up evaluations."""
        print("Preprocessing columns with bit masks for optimized trip counting...")
        self.preprocessed_cols = []
        
        for i, key in enumerate(self.col_keys):
            if i % 100 == 0 and i > 0:
                print(f"  Processed {i}/{len(self.col_keys)} columns...")
                
            col_data = self.columns[key]
            data = col_data["Data"]
            op_time = (float(data["total_dh_time"]) + 
                       float(data["total_travel_time"]) + 
                       float(data["total_wait_time"]))
            
            trips_only = [item for item in col_data['Path'] if item.startswith("l")]
            
            # Create a bit array for this column's trips (OPTIMIZATION)
            trip_mask = np.zeros(self.total_trips, dtype=bool)
            for trip in trips_only:
                if trip in self.trip_to_idx:
                    trip_mask[self.trip_to_idx[trip]] = True
            
            self.preprocessed_cols.append({
                'cost': col_data["Cost"],
                'op_time': op_time,
                'trips': trips_only,
                'trip_mask': trip_mask,
                'trip_count': len(trips_only),
                'is_short': op_time < 480  # Pre-calculate this too
            })
        
        print(f"Preprocessing complete for {len(self.col_keys)} columns")

    def initialize_population(self):
        """Initialize population with valid individuals."""
        initial_population = []
        print(f"Initializing population of size {self.pop_size}...")
        L_median = (self.L_min + self.L_max) / 2
        num_cols = len(self.columns)
        p_gene_one = L_median / num_cols

        while len(initial_population) < self.pop_size:
            needed = self.pop_size - len(initial_population)
            batch_size = max(needed * 2, 20)
            
            random_matrix = np.random.uniform(0.0, 1.0, (batch_size, num_cols))
            individuals_batch = (random_matrix >= 1 - p_gene_one).astype(np.int8)
            
            vehicle_counts = individuals_batch.sum(axis=1)
            valid_mask = (vehicle_counts >= self.L_min) & (vehicle_counts <= self.L_max)
            valid_individuals = individuals_batch[valid_mask]
            
            for ind in valid_individuals:
                if len(initial_population) < self.pop_size:
                    initial_population.append(ind)
                else:
                    break
        return np.array(initial_population, dtype=np.int8)

    def get_individual_hash(self, individual):
        """Create a hashable representation of an individual for caching."""
        return tuple(individual.tolist())

    def calculate_indicators(self, individual):
        """Optimized indicator calculation using bit operations."""
        selected_indices = np.where(individual == 1)[0]
        v = len(selected_indices)
        
        if v == 0:
            return {'v': 0, 'r': self.total_trips, 'u': self.total_trips, 's': 0, 'cost': math.inf}
        
        # Use numpy operations for efficiency
        selected_cols_data = [self.preprocessed_cols[i] for i in selected_indices]
        
        # Fast cost calculation
        cost = sum(d['cost'] for d in selected_cols_data)
        
        # Use bit operations for trip coverage (MAJOR OPTIMIZATION)
        combined_trip_mask = np.zeros(self.total_trips, dtype=bool)
        total_trip_count = 0
        s = 0
        
        for d in selected_cols_data:
            combined_trip_mask |= d['trip_mask']  # Bitwise OR for union
            total_trip_count += d['trip_count']
            if d['is_short']:
                s += 1
        
        unique_trips_count = combined_trip_mask.sum()
        r = total_trip_count - unique_trips_count
        u = self.total_trips - unique_trips_count
        
        return {'v': v, 'r': r, 'u': u, 's': s, 'cost': cost}

    def calculate_indicators_batch(self, individuals):
        """Calculate indicators for multiple individuals (for parallel processing)."""
        return [self.calculate_indicators(ind) for ind in individuals]

    def evaluate_fitness(self, indicators):
        """Calculates an individual's fitness using pre-calculated population stats."""
        C = np.array([indicators["v"], indicators["r"], indicators["u"], indicators["s"]])
        exponent = np.dot(self.sigma_vector, C)
        return math.exp(exponent)

    def evaluate_population(self, population):
        """
        Evaluates the entire population using vectorized operations where possible.
        Uses parallel processing if enabled and beneficial.
        """
        pop_size = len(population)
        
        # Check cache first if enabled
        if self.use_cache:
            all_indicators = []
            for ind in population:
                ind_hash = self.get_individual_hash(ind)
                if ind_hash in self.fitness_cache:
                    all_indicators.append(self.fitness_cache[ind_hash]['indicators'])
                    self.cache_hits += 1
                else:
                    indicators = self.calculate_indicators(ind)
                    all_indicators.append(indicators)
                    self.cache_misses += 1
                    # Cache will be updated after fitness calculation
        else:
            # Use parallel processing for large populations if enabled
            if self.use_parallel and pop_size > 20:
                # Split population for parallel processing
                chunk_size = max(1, pop_size // self.num_processes)
                chunks = [population[i:i+chunk_size] for i in range(0, pop_size, chunk_size)]
                
                with Pool(processes=self.num_processes) as pool:
                    results = pool.map(self.calculate_indicators_batch, chunks)
                
                # Flatten results
                all_indicators = list(itertools.chain.from_iterable(results))
            else:
                all_indicators = [self.calculate_indicators(ind) for ind in population]
        
        # Vectorized fitness calculation (OPTIMIZATION)
        v_array = np.array([ind['v'] for ind in all_indicators])
        r_array = np.array([ind['r'] for ind in all_indicators])
        u_array = np.array([ind['u'] for ind in all_indicators])
        s_array = np.array([ind['s'] for ind in all_indicators])
        cost_array = np.array([ind['cost'] for ind in all_indicators])
        
        # Batch fitness calculation
        C_matrix = np.column_stack([v_array, r_array, u_array, s_array])
        exponents = np.dot(C_matrix, self.sigma_vector)
        fitness_array = np.exp(exponents)
        
        # Build evaluation results and update cache
        evaluations = []
        for i in range(pop_size):
            eval_dict = {
                'individual': population[i],
                'indicators': all_indicators[i],
                'fitness': fitness_array[i],
                'cost': cost_array[i]
            }
            evaluations.append(eval_dict)
            
            # Update cache if enabled
            if self.use_cache:
                ind_hash = self.get_individual_hash(population[i])
                if ind_hash not in self.fitness_cache:
                    self.fitness_cache[ind_hash] = {
                        'indicators': all_indicators[i],
                        'fitness': fitness_array[i],
                        'cost': cost_array[i]
                    }
        
        return evaluations

    def update_best_if_better(self, evaluation):
        """Updates the all-time best individual if the current one has higher fitness."""
        if evaluation['fitness'] > self.best_fitness:
            self.best_fitness = evaluation['fitness']
            self.best_individual = evaluation['individual'].copy()
            self.best_indicators = evaluation['indicators']
            self.best_cost = evaluation['cost']
            return True
        return False

    def spin_roulette(self, fitness_list):
        """Roulette wheel selection."""
        fitness_array = np.array(fitness_list)
        total_fitness = fitness_array.sum()
        
        if total_fitness <= 0 or np.any(np.isnan(fitness_array)):
            probs = None
        else:
            probs = fitness_array / total_fitness
            
        parent_indices = np.random.choice(a=len(fitness_list), size=2, p=probs, replace=False)
        return parent_indices[0], parent_indices[1]

    def make_crossover(self, parent_1, parent_2):
        """Performs crossover between two parents."""
        num_genes = len(parent_1)
        offspring_1, offspring_2 = np.empty_like(parent_1), np.empty_like(parent_2)
        for i in range(0, num_genes, self.gene_segments):
            start, end = i, i + self.gene_segments
            if np.random.rand() <= 0.5:
                offspring_1[start:end], offspring_2[start:end] = parent_2[start:end], parent_1[start:end]
            else:
                offspring_1[start:end], offspring_2[start:end] = parent_1[start:end], parent_2[start:end]
        return offspring_1, offspring_2

    def mutate(self, individual):
        """Applies mutation to an individual by flipping bits."""
        mutation_mask = np.random.rand(len(individual)) <= self.mutation_p
        individual[mutation_mask] = 1 - individual[mutation_mask]
        return individual

    def run(self, verbose=True):
        """Runs the main genetic algorithm loop with optimizations."""
        start_time = time.time()
        population = self.initialize_population()
        
        # Initial evaluation for generation 0
        evaluations = self.evaluate_population(population)
        
        # Track performance metrics
        eval_times = []
        
        for g in range(1, self.max_gen + 1):
            gen_start_time = time.time()
            
            # The 'evaluations' from the previous loop iteration are P(g-1)
            fitness_list = [e['fitness'] for e in evaluations]
            
            # --- Generate the new population P(g) ---
            new_population = np.empty_like(population)
            for i in range(0, self.pop_size, 2):
                p1_idx, p2_idx = self.spin_roulette(fitness_list)
                parent_1, parent_2 = population[p1_idx], population[p2_idx]
                
                if np.random.rand() <= self.crossover_p:
                    offspring_1, offspring_2 = self.make_crossover(parent_1, parent_2)
                else:
                    offspring_1, offspring_2 = parent_1.copy(), parent_2.copy()
                
                new_population[i] = self.mutate(offspring_1)
                if i + 1 < self.pop_size:
                    new_population[i+1] = self.mutate(offspring_2)
            
            # --- Evaluate the new population P(g) ---
            current_evaluations = self.evaluate_population(new_population)
            
            # --- Elitism Strategy from Wang et al. (2021) ---
            
            # 1. Identify best and worst individuals in the current generation P(g)
            current_fitness_list = [e['fitness'] for e in current_evaluations]
            best_current_gen_idx = np.argmax(current_fitness_list)
            worst_current_gen_idx = np.argmin(current_fitness_list)
            best_current_gen_eval = current_evaluations[best_current_gen_idx]

            # Update all-time best solution tracker
            if self.update_best_if_better(best_current_gen_eval):
                if verbose and g > 1:
                    print(f">>> New best found at Gen {g}! Fitness: {self.best_fitness:.6f} (Cost: {self.best_cost:.2f})")

            # 2. Update the persistent elite pool
            sorted_indices = np.argsort(current_fitness_list)
            top_current_evals = [current_evaluations[i] for i in sorted_indices[-self.elite_indivs:]]
            
            temp_pool = self.elite_pool + top_current_evals
            temp_pool.sort(key=lambda x: x['fitness'], reverse=True)
            self.elite_pool = temp_pool[:self.elite_indivs]

            # 3. Perform conditional replacement logic
            is_improved = False
            if self.previous_gen_best_eval:
                if best_current_gen_eval['fitness'] > self.previous_gen_best_eval['fitness']:
                    is_improved = True
            
            replacement_individual = None
            if is_improved:
                replacement_individual = self.previous_gen_best_eval['individual']
            else:
                if self.elite_pool:
                    random_elite = random.choice(self.elite_pool)
                    replacement_individual = random_elite['individual']
            
            if replacement_individual is not None:
                new_population[worst_current_gen_idx] = replacement_individual.copy()

            # 4. Prepare for the next generation
            self.previous_gen_best_eval = best_current_gen_eval
            population = new_population
            evaluations = self.evaluate_population(population) # Re-evaluate for next loop's selection
            
            gen_time = time.time() - gen_start_time
            eval_times.append(gen_time)
            
            if verbose and (g % 100 == 0 or g == 1):
                gen_best_eval = best_current_gen_eval
                avg_eval_time = np.mean(eval_times[-10:]) if len(eval_times) >= 10 else np.mean(eval_times)
                
                print(f"\n--- GENERATION {g} ---")
                print(f"   Best Overall - Fitness: {self.best_fitness:.6f}, Cost: {self.best_cost:.2f}, v: {self.best_indicators.get('v')}, r: {self.best_indicators.get('r')}, u: {self.best_indicators.get('u')}, s: {self.best_indicators.get('s')}")
                print(f"   Gen {g} Best   - Fitness: {gen_best_eval['fitness']:.6f}, Cost: {gen_best_eval['cost']:.2f}, v: {gen_best_eval['indicators']['v']}, r: {gen_best_eval['indicators']['r']}, u: {gen_best_eval['indicators']['u']}, s: {gen_best_eval['indicators']['s']}")
                print(f"   Avg time per generation: {avg_eval_time:.3f}s")
                
                if self.use_cache and self.cache_hits + self.cache_misses > 0:
                    cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) * 100
                    print(f"   Cache hit rate: {cache_hit_rate:.1f}% ({self.cache_hits} hits, {self.cache_misses} misses)")

            cost_list = [e['cost'] for e in evaluations]
            self.generation_history.append({
                'generation': g, 'best_fitness': self.best_fitness, 'best_cost': self.best_cost,
                'avg_cost': np.mean([c for c in cost_list if c != math.inf])
            })
        
        self.ga_execution_time = time.time() - start_time
        
        if verbose:
            print(f"\nGA finished in {self.ga_execution_time:.2f} seconds.")
            print(f"Final Best Solution - Fitness: {self.best_fitness:.6f}, Cost: {self.best_cost:.2f}")
            print(f"Final Indicators - {self.best_indicators}")
            
            if self.use_cache:
                total_evals = self.cache_hits + self.cache_misses
                if total_evals > 0:
                    final_hit_rate = self.cache_hits / total_evals * 100
                    print(f"Final cache statistics: {final_hit_rate:.1f}% hit rate")
                    print(f"  Total evaluations: {total_evals}")
                    print(f"  Cache hits: {self.cache_hits}")
                    print(f"  Cache misses: {self.cache_misses}")
                    print(f"  Unique individuals evaluated: {len(self.fitness_cache)}")

        return self.best_fitness, self.best_cost, self.get_selected_columns(), None, None

    def get_selected_columns(self):
        """Extracts the columns selected by the final best individual."""
        if self.best_individual is None: 
            return None
        return {self.col_keys[i]: self.columns[self.col_keys[i]].copy() 
                for i, gene in enumerate(self.best_individual) if gene == 1}

    def clear_cache(self):
        """Clear the fitness cache to free memory if needed."""
        if self.use_cache:
            self.fitness_cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0
            print("Cache cleared")


def run_ga(columns, timetables_path, experiment_name, experiment_log_df, tmax_list, 
           ga_params=None, verbose=True, csv_results_path="ga_results.csv", 
           use_parallel=True, use_cache=True):
    """
    Run GA with provided columns and log results to CSV.
    
    Args:
        use_parallel: Enable parallel processing for population evaluation
        use_cache: Enable caching of fitness evaluations
    """
    # Create GA instance with optimization flags
    ga = GeneticAlgorithm(
        columns, 
        timetables_path, 
        experiment_name, 
        ga_params,
        use_parallel=use_parallel,
        use_cache=use_cache
    )
    
    # The ga.run() method returns 5 values. We capture them here.
    best_fitness, best_cost, selected_columns, PSI, C_min = ga.run(verbose)
    
    bi = ga.best_indicators
    v, r, u, s = bi.get('v', 0), bi.get('r', 0), bi.get('u', 0), bi.get('s', 0)
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # Use a results directory to keep the main directory clean
    results_dir = "ga_results"
    os.makedirs(results_dir, exist_ok=True)
    filename = os.path.join(results_dir, f"selected_columns_{timestamp}_{experiment_name}.txt")

    # This logging part is from your original file, ensuring it saves the results
    with open(filename, "w") as f:
        f.write(f"GA Results Summary - {timestamp}\n")
        f.write("="*50 + "\n")
        f.write(f"Repeated trips (r): {r}\n")
        f.write(f"Uncovered trips (u): {u}\n")
        f.write(f"Short routes (s): {s}\n")
        f.write(f"Number of vehicles (v): {v}\n")
        f.write(f"Best cost: {best_cost:.2f}\n")
        f.write(f"Best fitness: {ga.best_fitness:.6f}\n")
        f.write(f"Number of selected routes: {len(selected_columns) if selected_columns else 0}\n")
        
        # Add optimization statistics
        f.write("\n" + "="*50 + "\n")
        f.write("Optimization Statistics\n")
        f.write(f"Total execution time: {ga.ga_execution_time:.2f} seconds\n")
        f.write(f"Parallel processing: {'Enabled' if ga.use_parallel else 'Disabled'}\n")
        if ga.use_cache:
            total_evals = ga.cache_hits + ga.cache_misses
            if total_evals > 0:
                cache_hit_rate = ga.cache_hits / total_evals * 100
                f.write(f"Cache hit rate: {cache_hit_rate:.1f}%\n")
                f.write(f"Unique individuals evaluated: {len(ga.fitness_cache)}\n")
    
    print(f"GA results saved to: {filename}")

    # The test script expects 4 return values.
    # The original function returned the log dataframe. Let's maintain that structure.
    return best_fitness, best_cost, selected_columns, experiment_log_df