import pandas as pd
import numpy as np
import math
import time
import datetime
import os
import itertools
import random
import cProfile
import pstats

class GeneticAlgorithm:
    """Genetic Algorithm for column selection optimization."""
    
    def __init__(self, columns, timetables_path, experiment_name, ga_params=None):
        """Initialize GA with columns and parameters."""
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
        
        # Translation layer for backward compatibility
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
        
        self.sigma1 = params.get('sigma1', 0.0) 
        self.sigma2 = params.get('sigma2', 0.0) 
        self.sigma3 = params.get('sigma3', 0.0)
        self.sigma4 = params.get('sigma4', 0.0)
        
        if self.sigma1 == 0.0 and self.sigma2 == 0.0 and self.sigma3 == 0.0 and self.sigma4 == 0.0:
            print("\nWARNING: All fitness weights (sigma1, sigma2, sigma3, sigma4) are zero. All individuals will have a fitness of 1.0.")
            print("Please provide at least one sigma weight in ga_params for meaningful selection.\n")

        self.elite_pool = [] 
        self.previous_gen_best_eval = None

    def _preprocess_columns(self):
        """Pre-calculates static data into optimized NumPy arrays."""
        num_cols = len(self.col_keys)
        
        self.preprocessed_data = {
            'costs': np.zeros(num_cols),
            'op_times': np.zeros(num_cols),
            'trips': [set() for _ in range(num_cols)]
        }

        for i, key in enumerate(self.col_keys):
            col_data = self.columns[key]
            data = col_data["Data"]
            
            op_time = (float(data["total_dh_time"]) + 
                    float(data["total_travel_time"]) + 
                    float(data["total_wait_time"]))
            
            trips_only = {item for item in col_data['Path'] if item.startswith("l")}
            
            self.preprocessed_data['costs'][i] = col_data["Cost"]
            self.preprocessed_data['op_times'][i] = op_time
            self.preprocessed_data['trips'][i] = trips_only
         
    def initialize_population(self):
        """Initialize population with valid individuals."""
        initial_population = []
        print(f"Initializing population of size {self.pop_size}...")
        num_cols = len(self.columns)
        rng = np.random.default_rng()
        weights = np.array([1/num_cols] * num_cols)

        while len(initial_population) < self.pop_size:
            vehicles = rng.integers(self.L_min, self.L_max, endpoint=True)                
            individual_array = np.random.choice(num_cols, size=vehicles, replace=False, p=weights)

            mask = np.full(num_cols, True, dtype=bool)
            mask[individual_array] = False
            weights = np.empty(num_cols)
            weights[individual_array] = 0.2 / len(individual_array)
            weights[mask] = 0.8 / len(weights[mask])

            initial_population.append(np.sort(individual_array))

        return initial_population
    
    def calculate_indicators(self, individual_array):
        """Calculates performance indicators for a given individual."""
        v = len(individual_array)
        if v == 0:
            return {'v': 0, 'r': self.total_trips, 'u': self.total_trips, 's': 0, 'cost': math.inf}

        selected_costs = self.preprocessed_data['costs'][individual_array]
        selected_op_times = self.preprocessed_data['op_times'][individual_array]
        
        cost = np.sum(selected_costs)
        s = np.sum(selected_op_times < 480)
        
        selected_trips = [self.preprocessed_data['trips'][i] for i in individual_array]
        total_trip_occurrences = sum(map(len, selected_trips))
        unique_trips = set(itertools.chain.from_iterable(selected_trips))
            
        unique_trips_count = len(unique_trips)
        r = total_trip_occurrences - unique_trips_count
        u = self.total_trips - unique_trips_count

        return {'v': v, 'r': r, 'u': u, 's': s, 'cost': cost}

    def evaluate_fitness(self, indicators):
        """Calculates an individual's fitness."""
        Sigma = np.array([self.sigma1, self.sigma2, self.sigma3, self.sigma4]) * -1
        C = [indicators["v"], indicators["r"], indicators["u"], indicators["s"]]
        exponent = np.dot(Sigma, C)
        return math.exp(exponent)
    
    def evaluate_population_vectorized(self, population):
        """Evaluates the entire population using vectorized fitness calculation."""
        pop_size = len(population)
        
        v_values = np.zeros(pop_size, dtype=int)
        r_values = np.zeros(pop_size, dtype=int)
        u_values = np.zeros(pop_size, dtype=int)
        s_values = np.zeros(pop_size, dtype=int)
        cost_values = np.zeros(pop_size, dtype=float)

        for i, individual in enumerate(population):
            indicators = self.calculate_indicators(individual)
            v_values[i] = indicators['v']
            r_values[i] = indicators['r']
            u_values[i] = indicators['u']
            s_values[i] = indicators['s']
            cost_values[i] = indicators['cost']
            
        indicator_matrix = np.vstack([v_values, r_values, u_values, s_values]).T
        Sigma = np.array([self.sigma1, self.sigma2, self.sigma3, self.sigma4]) * -1
        exponents = np.dot(indicator_matrix, Sigma)
        fitness_values = np.exp(exponents)

        evaluations = [
            {
                'individual': population[i],
                'indicators': {
                    'v': v_values[i], 'r': r_values[i], 'u': u_values[i], 
                    's': s_values[i], 'cost': cost_values[i]
                },
                'fitness': fitness_values[i],
                'cost': cost_values[i]
            } for i in range(pop_size)
        ]
        
        return evaluations

    def update_best_if_better(self, evaluation):
        """Updates the all-time best individual if current has higher fitness."""
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
    
    def segment_individual(self, individual_array):
        """Segments an individual array by finding split points."""
        sorted_array = np.sort(individual_array)
        num_columns = len(self.columns)
        step = self.gene_segments
        boundaries = np.arange(0, num_columns + step, step)
        boundaries[-1] = num_columns

        split_indices = np.searchsorted(sorted_array, boundaries, side='right')
        
        segmented_individual = [
            sorted_array[split_indices[i]:split_indices[i+1]]
            for i in range(len(split_indices) - 1)
        ]

        return segmented_individual

    def make_crossover(self, parent_1, parent_2):
        """Performs vectorized crossover between two parents."""
        segmented_parent_1 = self.segment_individual(parent_1)
        segmented_parent_2 = self.segment_individual(parent_2)
        
        num_segments = len(segmented_parent_1)
        crossover_mask = np.random.rand(num_segments) > 0.5
        
        segs_for_offspring_1 = [
            segmented_parent_1[i] if crossover_mask[i] else segmented_parent_2[i]
            for i in range(num_segments)
        ]
        segs_for_offspring_2 = [
            segmented_parent_2[i] if crossover_mask[i] else segmented_parent_1[i]
            for i in range(num_segments)
        ]
        
        offspring_1 = np.concatenate(segs_for_offspring_1) if segs_for_offspring_1 else np.array([])
        offspring_2 = np.concatenate(segs_for_offspring_2) if segs_for_offspring_2 else np.array([])

        offspring_1_sorted = np.sort(offspring_1)
        offspring_2_sorted = np.sort(offspring_2)
        
        return offspring_1_sorted, offspring_2_sorted

    def mutate(self, individual_array):
        """Mutates an individual using symmetric difference."""
        mutation_mask = np.random.rand(len(self.columns)) <= self.mutation_p
        indexes_for_mutation = np.where(mutation_mask)[0]
        mutated_array = np.setxor1d(individual_array, indexes_for_mutation, assume_unique=True)
        return mutated_array

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
        
        evaluations = self.evaluate_population_vectorized(population) 
        
        if evaluations:
            initial_best_eval = max(evaluations, key=lambda x: x['fitness'])
            self.update_best_if_better(initial_best_eval)

        if verbose:
            print(f"Initial best: fitness={self.best_fitness:.6f}, cost={self.best_cost:.2f}")
        
        for g in range(1, self.max_gen + 1):
            fitness_list = [e['fitness'] for e in evaluations]
            
            if verbose and (g % 100 == 0 or g == 1):
                print(f"\nGENERATION {g} " + "-"*50)
                print(f"Best overall - Fitness: {self.best_fitness:.6f}, Cost: {self.best_cost:.2f}, v: {self.best_indicators.get('v')}, r: {self.best_indicators.get('r')}, u: {self.best_indicators.get('u')}, s: {self.best_indicators.get('s')}")
            
            new_population = [None] * self.pop_size
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
            
            current_evaluations = self.evaluate_population_vectorized(new_population)

            # Elitism strategy
            current_fitness_list = [e['fitness'] for e in current_evaluations]
            best_current_gen_idx = np.argmax(current_fitness_list)
            worst_current_gen_idx = np.argmin(current_fitness_list)
            best_current_gen_eval = current_evaluations[best_current_gen_idx]

            if self.update_best_if_better(best_current_gen_eval):
                if verbose and g > 1:
                    print(f">>> New best found at Gen {g}! Fitness: {self.best_fitness:.6f} (Cost: {self.best_cost:.2f})")

            sorted_indices = np.argsort(current_fitness_list)
            top_current_evals = [current_evaluations[i] for i in sorted_indices[-self.elite_indivs:]]
            
            temp_pool = self.elite_pool + top_current_evals
            temp_pool.sort(key=lambda x: x['fitness'], reverse=True)
            self.elite_pool = temp_pool[:self.elite_indivs]

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

                indicators = self.calculate_indicators(replacement_individual)
                fitness = self.evaluate_fitness(indicators)
                
                current_evaluations[worst_current_gen_idx] = {
                    'individual': replacement_individual,
                    'indicators': indicators,
                    'fitness': fitness,
                    'cost': indicators['cost']
                }

            self.previous_gen_best_eval = best_current_gen_eval
            population = new_population
            evaluations = current_evaluations
            
            if verbose and (g % 100 == 0 or g == 1):
                gen_best_eval = best_current_gen_eval
                print(f"Gen {g} best   - Fitness: {gen_best_eval['fitness']:.6f}, Cost: {gen_best_eval['cost']:.2f}, v: {gen_best_eval['indicators']['v']}, r: {gen_best_eval['indicators']['r']}, u: {gen_best_eval['indicators']['u']}, s: {gen_best_eval['indicators']['s']}")

            cost_list = [e['cost'] for e in evaluations]
            self.generation_history.append({
                'generation': g, 'best_fitness': self.best_fitness, 'best_cost': self.best_cost,
                'avg_cost': np.mean([c for c in cost_list if c != math.inf]) if cost_list else 0
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
        
        selected_indices = self.best_individual
        selected_columns = {}
        route_counter = 0
        
        for col_index in selected_indices:
            original_key = self.col_keys[col_index]
            new_key = f"Route_{route_counter}"
            selected_columns[new_key] = self.columns[original_key].copy()
            route_counter += 1
                
        return selected_columns


def run_ga(columns, timetables_path, experiment_name, ga_params=None, verbose=True):
    """Run GA with provided columns and log results to CSV."""
    #profiler = cProfile.Profile()
    
    ga = GeneticAlgorithm(columns, timetables_path, experiment_name, ga_params)
    
    #profiler.enable()
    best_fitness, best_cost, selected_columns, PSI, C_min = ga.run(verbose)
    #profiler.disable()
    
    #stats = pstats.Stats(profiler).sort_stats('cumulative')
    #stats.print_stats(20)
    ga_execution_log = ga.log_ga_execution()
    
    bi = ga.best_indicators
    v, r, u, s = bi.get('v', 0), bi.get('r', 0), bi.get('u', 0), bi.get('s', 0)
    
    if verbose:
        print(f"Summary: {v} vehicles, {r} repeated trips, {u} uncovered trips, {s} short routes")
    
    return best_fitness, best_cost, v, r, u, s, selected_columns, ga_execution_log