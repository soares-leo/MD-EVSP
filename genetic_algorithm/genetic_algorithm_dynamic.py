import pandas as pd
import numpy as np
import math
import time
import datetime
import os
import itertools

class GeneticAlgorithm:
    """
    Genetic Algorithm for column selection optimization.
    
    Uses a dynamic exponential fitness scaling function based on Lima et al. (1996).
    The primary goal is always to MAXIMIZE fitness.
    This version includes a backward-compatibility adapter for older parameter formats.
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
        # This block adapts the old parameter format (phi_v, phi_u, etc.) 
        # to the new, more flexible format (fitness_components, phi_values).
        if 'fitness_components' in params and 'phi_values' in params:
            # If new format is provided, use it directly.
            print("Using new GA parameter format (fitness_components, phi_values).")
            self.fitness_components = params['fitness_components']
            self.phi_values = params['phi_values']
        else:
            # If new format is not found, try to translate from the old format.
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
                # If no translatable parameters found, use a hardcoded default.
                print("No valid old or new parameters found. Using default fitness settings.")
                self.fitness_components = ['v', 'r', 'u', 's']
                self.phi_values = {'v': 0.01, 'r': 0.01, 'u': 0.8, 's': 0.2}

        print(f"Fitness will be based on components: {self.fitness_components} with phi values: {self.phi_values}")

        # Load timetables and calculate constraints
        self.timetable_trips = pd.read_csv(timetables_path, usecols=["trip_id"])
        num_of_trips = len(self.timetable_trips)
        self.total_trips = num_of_trips
        self.L_min = int(round((self.L_min_constant / num_of_trips) * num_of_trips))
        self.L_max = int(round((self.L_max_constant / num_of_trips) * num_of_trips))

        self._preprocess_columns()
        
        # Results storage - best is always determined by MAX fitness
        self.best_individual = None
        self.best_fitness = -math.inf
        self.best_cost = math.inf
        self.best_indicators = {}
        
        self.generation_history = []

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

    def evaluate_fitness(self, indicators, C_min, PSI):
        """Calculates an individual's fitness using pre-calculated population stats."""
        exponent = 0.0
        for component in self.fitness_components:
            C = indicators.get(component, 0)
            C_min_val = C_min.get(component, 0)
            psi_val = PSI.get(component, 0)
            exponent += psi_val * (C - C_min_val)
        return math.exp(exponent)

    def evaluate_population(self, population):
        """
        Evaluates the entire population using the dynamic exponential fitness function.
        """
        all_indicators = [self.calculate_indicators(ind) for ind in population]
        
        C_min, C_avg = {}, {}
        for component in self.fitness_components:
            values = [ind[component] for ind in all_indicators]
            C_min[component] = np.min(values)
            C_avg[component] = np.mean(values)

        PSI = {}
        for component in self.fitness_components:
            diff = C_avg[component] - C_min[component]
            phi = self.phi_values.get(component, 0.5)
            PSI[component] = math.log(phi) / (diff + 1e-6)

        evaluations = []
        for i, ind in enumerate(population):
            indicators = all_indicators[i]
            fitness = self.evaluate_fitness(indicators, C_min, PSI)
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

    def run(self, verbose=True):
        """Runs the main genetic algorithm loop."""
        start_time = time.time()
        population = self.initialize_population()
        
        for g in range(1, self.max_gen + 1):
            evaluations = self.evaluate_population(population)
            fitness_list = [e['fitness'] for e in evaluations]
            
            best_gen_idx = np.argmax(fitness_list)
            if self.update_best_if_better(evaluations[best_gen_idx]):
                if verbose and g > 1:
                    print(f">>> New best found at Gen {g}! Fitness: {self.best_fitness:.6f} (Cost: {self.best_cost:.2f})")

            elite_indices = np.argsort(fitness_list)[-self.elite_indivs:]
            elite_individuals = [population[i].copy() for i in elite_indices]
            
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

            new_evals = self.evaluate_population(new_population)
            new_fitness_list = [e['fitness'] for e in new_evals]
            worst_new_indices = np.argsort(new_fitness_list)[:self.elite_indivs]

            for i, elite_ind in enumerate(elite_individuals):
                new_population[worst_new_indices[i]] = elite_ind

            population = new_population

            if verbose and (g % 100 == 0 or g == 1):
                gen_best_eval = evaluations[best_gen_idx]
                print(f"\n--- GENERATION {g} ---")
                print(f"  Best Overall - Fitness: {self.best_fitness:.6f}, Cost: {self.best_cost:.2f}, v: {self.best_indicators.get('v')}, r: {self.best_indicators.get('r')}, u: {self.best_indicators.get('u')}, s: {self.best_indicators.get('s')}")
                print(f"  Gen {g} Best  - Fitness: {gen_best_eval['fitness']:.6f}, Cost: {gen_best_eval['cost']:.2f}, v: {gen_best_eval['indicators']['v']}, r: {gen_best_eval['indicators']['r']}, u: {gen_best_eval['indicators']['u']}, s: {gen_best_eval['indicators']['s']}")

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

        return 1.0, self.best_cost, self.get_selected_columns(), None, None

    def get_selected_columns(self):
        """Extracts the columns selected by the final best individual."""
        if self.best_individual is None: return None
        return {self.col_keys[i]: self.columns[self.col_keys[i]].copy() for i, gene in enumerate(self.best_individual) if gene == 1}

# This convenience function is called by your script. It doesn't need changes.
def run_ga(columns, timetables_path, experiment_name, experiment_log_df, tmax_list, ga_params=None, verbose=True, csv_results_path="ga_results.csv"):
    """
    Run GA with provided columns and log results to CSV.
    """
    ga = GeneticAlgorithm(columns, timetables_path, experiment_name, ga_params)
    best_fitness, best_cost, selected_columns, PSI, C_min = ga.run(verbose)
    
    bi = ga.best_indicators
    v, r, u, s = bi.get('v', 0), bi.get('r', 0), bi.get('u', 0), bi.get('s', 0)
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"selected_columns_{timestamp}_{experiment_name}.txt"

    with open(filename, "w") as f:
        f.write(f"GA Results Summary - {timestamp}\n")
        f.write("="*50 + "\n")
        f.write(f"Repeated trips (r): {r}\n")
        f.write(f"Uncovered trips (u): {u}\n")
        f.write(f"Short routes (s): {s}\n")
        f.write(f"Number of vehicles (v): {v}\n")
        f.write(f"Best cost: {best_cost:.2f}\n")
        f.write(f"Best fitness: {ga.best_fitness:.6f}\n")
        # Add a check for selected_columns being None
        f.write(f"Number of selected routes: {len(selected_columns) if selected_columns else 0}\n")

    # The rest of the logging can proceed as before.
    return best_fitness, best_cost, selected_columns, experiment_log_df