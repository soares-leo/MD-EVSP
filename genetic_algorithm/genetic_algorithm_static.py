import pandas as pd
import numpy as np
import math
import time
import datetime
import os
import itertools
import random

class GeneticAlgorithm:
    """
    Genetic Algorithm for column selection optimization.
    
    Uses a dynamic exponential fitness scaling function based on Lima et al. (1996).
    The primary goal is always to MAXIMIZE fitness.
    This version includes a backward-compatibility adapter for older parameter formats.
    
    MODIFIED: Elitism strategy now aligns with Wang et al. (2021) paper.
    """
    
    def __init__(self, columns, timetables_path, experiment_name, ga_params=None):
        """
        Initialize GA with columns and parameters.
        Includes a translation layer for backward compatibility with older ga_params formats.
        """
        self.columns = columns
        self.col_keys = list(columns.keys())
        self.timetables_path = timetables_path
        self.experiment_name = experiment_name
        
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

        self.timetable_trips = pd.read_csv(timetables_path, usecols=["trip_id"])
        num_of_trips = len(self.timetable_trips)
        self.total_trips = num_of_trips
        self.L_min = int(round((self.L_min_constant / num_of_trips) * num_of_trips))
        self.L_max = int(round((self.L_max_constant / num_of_trips) * num_of_trips))

        self._preprocess_columns()
        
        self.best_individual = None
        self.best_fitness = -math.inf
        self.best_cost = math.inf
        self.best_indicators = {}
        
        self.generation_history = []
        
        # --- MODIFIED SECTION (as per user request) ---
        # This section now initializes the four sigma parameters.
        # sigma1 is applied to v
        self.sigma1 = params.get('sigma1', 0.0) 
        # sigma2 is applied to r
        self.sigma2 = params.get('sigma2', 0.0) 
        # sigma3 is applied to u
        self.sigma3 = params.get('sigma3', 0.0)
        # sigma4 is applied to s
        self.sigma4 = params.get('sigma4', 0.0)
        
        # Add a warning if no sigma weights are specified, as this would make selection random.
        if self.sigma1 == 0.0 and self.sigma2 == 0.0 and self.sigma3 == 0.0 and self.sigma4 == 0.0:
            print("\nWARNING: All fitness weights (sigma1, sigma2, sigma3, sigma4) are zero. All individuals will have a fitness of 1.0.")
            print("Please provide at least one sigma weight in ga_params for meaningful selection.\n")
        # --- END OF MODIFIED SECTION ---

        # Attributes for paper's elitism strategy
        self.elite_pool = [] 
        self.previous_gen_best_eval = None

    def _preprocess_columns(self):
        """Pre-calculates static data for each column to speed up evaluations."""
        self.preprocessed_cols = []
        for key in self.col_keys:
            col_data = self.columns[key]
            data = col_data["Data"]
            op_time = (float(data["total_dh_time"]) + 
                       float(data["total_travel_time"]) + 
                       float(data["total_wait_time"]))
            
            trips_only = [item for item in col_data['Path'] if item.startswith("l")]
            
            self.preprocessed_cols.append({
                'cost': col_data["Cost"],
                'op_time': op_time,
                'trips': trips_only
            })

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

    def calculate_indicators(self, individual):
        """Calculates all performance indicators for a single individual."""
        selected_indices = np.where(individual == 1)[0]
        v = len(selected_indices)

        if v == 0:
            return {'v': 0, 'r': self.total_trips, 'u': self.total_trips, 's': 0, 'cost': math.inf}

        selected_cols_data = [self.preprocessed_cols[i] for i in selected_indices]
        cost = sum(d['cost'] for d in selected_cols_data)
        operation_times = [d['op_time'] for d in selected_cols_data]
        flat_trips = list(itertools.chain.from_iterable(d['trips'] for d in selected_cols_data))
        unique_trips_count = len(set(flat_trips))
        
        r = len(flat_trips) - unique_trips_count
        u = self.total_trips - unique_trips_count
        s = sum(1 for t in operation_times if t < 480)

        return {'v': v, 'r': r, 'u': u, 's': s, 'cost': cost}

    def evaluate_fitness(self, indicators):
        """Calculates an individual's fitness using pre-calculated population stats."""
        # MODIFIED: Using four sigma parameters for v, r, u, and s respectively.
        Sigma = np.array([self.sigma1, self.sigma2, self.sigma3, self.sigma4]) * -1
        
        C = [indicators["v"], indicators["r"], indicators["u"], indicators["s"]]
        exponent = np.dot(Sigma, C)
        return math.exp(exponent)

    def evaluate_population(self, population):
        """
        Evaluates the entire population using the dynamic exponential fitness function.
        """
        all_indicators = [self.calculate_indicators(ind) for ind in population]

        evaluations = []
        for i, ind in enumerate(population):
            indicators = all_indicators[i]
            fitness = self.evaluate_fitness(indicators)
            evaluations.append({
                'individual': ind,
                'indicators': indicators,
                'fitness': fitness,
                'cost': indicators['cost']
            })
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

    def log_ga_execution(self):
        """Log execution metrics to experiment dataframe."""
        bi = self.best_indicators
        v, r, u, s = bi.get('v', 0), bi.get('r', 0), bi.get('u', 0), bi.get('s', 0)
        
        metrics = ["ga_execution_time", "best_cost", "best_fitness", "best_v", "best_r", "best_u", "best_s"]
        log_ids = [f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}{i}" for i in range(len(metrics))]
        values = [self.ga_execution_time, self.best_cost, self.best_fitness, v, r, u, s]
        
        df = pd.DataFrame({
            "log_ids": log_ids,
            "experiment_name": [self.experiment_name] * len(metrics),
            "step": ["genetic_algorithm"] * len(metrics),
            "metric": metrics,
            "value": values,
        })

        return df

    def run(self, verbose=True):
        """Runs the main genetic algorithm loop."""
        start_time = time.time()
        population = self.initialize_population()
        
        # Initial evaluation for generation 0
        evaluations = self.evaluate_population(population)
        
        if verbose:
            print(f"Initial best: fitness={self.best_fitness:.6f}, cost={self.best_cost:.2f}")
        
        for g in range(1, self.max_gen + 1):
            # The 'evaluations' from the previous loop iteration are P(g-1)
            fitness_list = [e['fitness'] for e in evaluations]
            
            if verbose and (g % 100 == 0 or g == 1):
                print(f"\nGENERATION {g} " + "-"*50)
                print(f"Best overall - Fitness: {self.best_fitness:.6f}, Cost: {self.best_cost:.2f}, v: {self.best_indicators.get('v')}, r: {self.best_indicators.get('r')}, u: {self.best_indicators.get('u')}, s: {self.best_indicators.get('s')}")
            
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
            
            if verbose and (g % 100 == 0 or g == 1):
                gen_best_eval = best_current_gen_eval
                print(f"Gen {g} best   - Fitness: {gen_best_eval['fitness']:.6f}, Cost: {gen_best_eval['cost']:.2f}, v: {gen_best_eval['indicators']['v']}, r: {gen_best_eval['indicators']['r']}, u: {gen_best_eval['indicators']['u']}, s: {gen_best_eval['indicators']['s']}")

            cost_list = [e['cost'] for e in evaluations]
            self.generation_history.append({
                'generation': g, 'best_fitness': self.best_fitness, 'best_cost': self.best_cost,
                'avg_cost': np.mean([c for c in cost_list if c != math.inf])
            })
        
        self.ga_execution_time = time.time() - start_time
        if verbose:
            print(f"\nGA completed after {self.max_gen} generations")
            print(f"Final best fitness={self.best_fitness:.6f} (Cost={self.best_cost:.2f})")
            print(f"Final best indicators - v: {self.best_indicators.get('v')}, r: {self.best_indicators.get('r')}, u: {self.best_indicators.get('u')}, s: {self.best_indicators.get('s')}")
            print(f"Execution time: {self.ga_execution_time:.2f} seconds")

        return self.best_fitness, self.best_cost, self.get_selected_columns(), None, None

    def get_selected_columns(self):
        """Extracts the columns selected by the final best individual."""
        if self.best_individual is None: 
            return None
        
        selected_columns = {}
        route_counter = 0
        
        for i, gene in enumerate(self.best_individual):
            if gene == 1:
                original_key = self.col_keys[i]
                new_key = f"Route_{route_counter}"
                selected_columns[new_key] = self.columns[original_key].copy()
                route_counter += 1
                
        return selected_columns


def run_ga(columns, timetables_path, experiment_name, ga_params=None, verbose=True):
    """
    Run GA with provided columns and log results to CSV.
    """
    ga = GeneticAlgorithm(columns, timetables_path, experiment_name, ga_params)
    
    # The ga.run() method returns 5 values. We capture them here.
    best_fitness, best_cost, selected_columns, PSI, C_min = ga.run(verbose)
    ga_execution_log = ga.log_ga_execution()
    
    bi = ga.best_indicators
    v, r, u, s = bi.get('v', 0), bi.get('r', 0), bi.get('u', 0), bi.get('s', 0)
    
    if verbose:
        print(f"Summary: {v} vehicles, {r} repeated trips, {u} uncovered trips, {s} short routes")
    
    return best_fitness, best_cost, v, r, u, s, selected_columns, ga_execution_log