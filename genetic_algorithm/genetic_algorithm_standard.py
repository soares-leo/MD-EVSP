import pandas as pd
import numpy as np
import math
import time
import datetime
import json
import os
import itertools # Import itertools for efficient iteration

class GeneticAlgorithm:
    """
    Genetic Algorithm for column selection optimization.
    Uses dynamic selectivity scaling function based on Lima et al. (1996).
    """
    
    def __init__(self, columns, timetables_path, experiment_name, ga_params=None):
        """
        Initialize GA with columns and parameters.
        
        Args:
            columns: Dictionary of columns from column generation or initial solution
            timetables_path: Path to timetables CSV file
            ga_params: Dictionary of GA parameters (optional)
        """
        self.columns = columns
        self.col_keys = list(columns.keys())
        self.timetables_path = timetables_path
        self.experiment_name = experiment_name
        
        # Set GA parameters (use defaults if not provided)
        params = ga_params or {}
        self.max_gen = params.get('max_gen', 1000)
        self.crossover_p = params.get('crossover_p', 0.3)
        self.mutation_p = params.get('mutation_p', 0.01)
        self.pop_size = params.get('pop_size', 100)
        self.elite_indivs = params.get('elite_indivs', 5)
        
        self.w1 = params.get('w1', 1)
        self.w2 = params.get('w2', 1)
        self.w3 = params.get('w3', 1)
        self.w4 = params.get('w4', 1)
        
        self.gene_segments = params.get('gene_segments', 20)
        self.L_min_constant = params.get('L_min_constant', 140)
        self.L_max_constant = params.get('L_max_constant', 180)
        
        # Load timetables and create required trips set
        self.timetable_trips = pd.read_csv(timetables_path, usecols=["trip_id"])
        
        num_of_trips = len(self.timetable_trips)
        L_min_fraction_constant = self.L_min_constant / num_of_trips
        L_max_fraction_constant = self.L_max_constant / num_of_trips

        self.L_min = int(round(L_min_fraction_constant * num_of_trips))
        self.L_max = int(round(L_max_fraction_constant * num_of_trips))

        self.total_trips = len(self.timetable_trips["trip_id"].values)
        
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
        
        self.best_individual = None
        self.best_cost = math.inf
        self.best_fitness = -math.inf # Fitness is maximized, so start at negative infinity
        self.best_v = math.inf
        self.best_r = math.inf
        self.best_u = math.inf
        self.best_ru = math.inf
        self.best_s = math.inf
        
        self.generation_history = []
        self.use_objective = ga_params.get('use_objective', True)

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
        
        return initial_population

    def calculate_indicators(self, individual):
        """
        Calculate indicators and other useful data for further evaluation.
        """
        selected_indices = np.where(individual == 1)[0]
        v = len(selected_indices)

        if v == 0:
            cost = math.inf
            return 0, 0, self.total_trips, 0, cost, []

        selected_cols_data = [self.preprocessed_cols[i] for i in selected_indices]
        
        cost = sum(d['cost'] for d in selected_cols_data)
        operation_times = [d['op_time'] for d in selected_cols_data]
        
        trips_from_selected_cols = (d['trips'] for d in selected_cols_data)
        flat_trips = list(itertools.chain.from_iterable(trips_from_selected_cols))
        
        unique_trips_count = len(set(flat_trips))
        
        r = len(flat_trips) - unique_trips_count
        u = self.total_trips - unique_trips_count
        s = sum(1 for t in operation_times if t < 480)

        return v, r, u, s, cost, flat_trips
    
    def evaluate_fitness(self, v, u, r, s):
        """
        Calculate fitness using dynamic selectivity scaling function.
        """
        denominator = (self.w1 * v + self.w2 * (r + u) + self.w3 * s)
        if denominator == 0:
            return math.inf 
        fitness = self.w4 / denominator
        return fitness

    def evaluate_population(self, population):
        """
        Evaluate entire population, calculating all metrics and fitness.
        """
        indicators_list = [self.calculate_indicators(ind) for ind in population]
        
        v_list = [item[0] for item in indicators_list]
        r_list = [item[1] for item in indicators_list]
        u_list = [item[2] for item in indicators_list]
        s_list = [item[3] for item in indicators_list]
        cost_list = [item[4] for item in indicators_list]
        
        if not self.use_objective:
            fitness_list = [
                self.evaluate_fitness(v, u, r, s)
                for v,r,u,s in zip(v_list, r_list, u_list, s_list)
            ]
        else:
            fitness_list = cost_list.copy()
            
        return fitness_list, cost_list, v_list, r_list, u_list, s_list

    def spin_roulette(self, fitness_list, population):
        """Roulette wheel selection for parent selection."""
        fitness_array = np.array(fitness_list)

        if self.use_objective:
            max_fit = np.max(fitness_array[np.isfinite(fitness_array)])
            if not np.isfinite(max_fit): # Handle case where all are inf
                fitness_array = np.ones_like(fitness_array)
            else:
                fitness_array = (max_fit + 1) - fitness_array
                fitness_array[~np.isfinite(fitness_array)] = 0 # Set inf costs to 0 probability


        total_fitness = fitness_array.sum()
        
        if total_fitness <= 0 or np.any(np.isnan(fitness_array)):
            probs = None
        else:
            probs = fitness_array / total_fitness
            
        parent_indices = np.random.choice(
            a=len(population), size=2, p=probs, replace=False
        )
        
        return population[parent_indices[0]], population[parent_indices[1]]

    def make_crossover(self, parents):
        """Perform crossover between two parents."""
        parent_1, parent_2 = parents
        num_genes = len(parent_1)
        
        offspring_1 = np.empty(num_genes, dtype=np.int8)
        offspring_2 = np.empty(num_genes, dtype=np.int8)

        for i in range(0, num_genes, self.gene_segments):
            start, end = i, i + self.gene_segments
            
            if np.random.uniform(0.0, 1.0) <= 0.5:
                offspring_1[start:end] = parent_2[start:end]
                offspring_2[start:end] = parent_1[start:end]
            else:
                offspring_1[start:end] = parent_1[start:end]
                offspring_2[start:end] = parent_2[start:end]
                
        return offspring_1, offspring_2

    def mutate(self, individual):
        """Apply mutation to an individual."""
        mutation_mask = np.random.uniform(0.0, 1.0, len(individual)) <= self.mutation_p
        individual[mutation_mask] = 1 - individual[mutation_mask]
        return individual

    def update_best_if_better(self, individual, fitness, cost, v, r, u, s):
        """
        Update best individual if the given one is better.
        - If use_objective is True, "better" means lower cost.
        - If use_objective is False, "better" means higher fitness.
        """
        is_new_best = False
        if self.use_objective:
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_fitness = fitness # Store corresponding fitness
                is_new_best = True
        else:
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_cost = cost # Store corresponding cost
                is_new_best = True
        
        if is_new_best:
            self.best_individual = individual.copy()
            self.best_v = v
            self.best_r = r
            self.best_u = u
            self.best_ru = r + u
            self.best_s = s
        
        return is_new_best

    def get_selected_columns(self):
        """Extract the columns selected by the best individual."""
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

    def run(self, verbose=True):
        """Run the genetic algorithm."""
        start_time = time.time()
        
        # Initialize population and evaluate
        population = np.array(self.initialize_population(), dtype=np.int8)
        fitness_list, cost_list, v_list, r_list, u_list, s_list = self.evaluate_population(population)
        
        if self.use_objective:
            best_gen_idx = np.argmin(fitness_list) # fitness_list is cost_list
        else:
            best_gen_idx = np.argmax(fitness_list)
            
        self.update_best_if_better(
            population[best_gen_idx], fitness_list[best_gen_idx], cost_list[best_gen_idx], 
            v_list[best_gen_idx], r_list[best_gen_idx], u_list[best_gen_idx], s_list[best_gen_idx]
        )
        if verbose:
            print(f"Initial best: cost={self.best_cost:.2f}, fitness={self.best_fitness:.4f}, v={self.best_v}, r={self.best_r}, u={self.best_u}, s={self.best_s}")

        for g in range(1, self.max_gen + 1):
            if verbose and (g % 100 == 0 or g == 1):
                print(f"\nGENERATION {g} " + "-"*50)
                # --- MODIFIED: Print all indicators for the best overall solution ---
                print(f"Best overall - Fitness: {self.best_fitness:.4f}, Cost: {self.best_cost:.2f}, v: {self.best_v}, r: {self.best_r}, u: {self.best_u}, s: {self.best_s}")
            
            # Elitism: Identify the best individuals from the current population based on FITNESS
            if self.use_objective:
                elite_indices = np.argsort(fitness_list)[:self.elite_indivs]
            else:
                elite_indices = np.argsort(fitness_list)[-self.elite_indivs:]
            
            elite_population = [population[i].copy() for i in elite_indices]
            
            # Generate New Population
            new_population = np.empty_like(population)
            for i in range(0, self.pop_size, 2):
                parent_1, parent_2 = self.spin_roulette(fitness_list, population)
                
                if np.random.uniform(0.0, 1.0) <= self.crossover_p:
                    offspring_1, offspring_2 = self.make_crossover((parent_1, parent_2))
                else:
                    offspring_1, offspring_2 = parent_1.copy(), parent_2.copy()
                
                new_population[i] = self.mutate(offspring_1)
                if i + 1 < self.pop_size:
                    new_population[i+1] = self.mutate(offspring_2)
            
            # Replace worst new individuals with elites
            new_fitness_list, _, _, _, _, _ = self.evaluate_population(new_population)
            
            if self.use_objective:
                worst_new_indices = np.argsort(new_fitness_list)[-self.elite_indivs:]
            else:
                worst_new_indices = np.argsort(new_fitness_list)[:self.elite_indivs]

            for i, elite_ind in enumerate(elite_population):
                new_population[worst_new_indices[i]] = elite_ind.copy()

            population = new_population
            fitness_list, cost_list, v_list, r_list, u_list, s_list = self.evaluate_population(population)

            if self.use_objective:
                current_best_idx = np.argmin(fitness_list)
            else:
                current_best_idx = np.argmax(fitness_list)
            
            # Gather all data for the current generation's best
            current_fitness = fitness_list[current_best_idx]
            current_cost = cost_list[current_best_idx]
            current_v = v_list[current_best_idx]
            current_r = r_list[current_best_idx]
            current_u = u_list[current_best_idx]
            current_s = s_list[current_best_idx]

            if self.update_best_if_better(
                population[current_best_idx], current_fitness, current_cost,
                current_v, current_r, current_u, current_s
            ):
                if verbose and g > 1:
                    if self.use_objective:
                        print(f">>> New best found at Gen {g}! Cost: {self.best_cost:.2f}")
                    else:
                        print(f">>> New best found at Gen {g}! Fitness: {self.best_fitness:.4f} (Cost: {self.best_cost:.2f})")
            
            if verbose and (g % 100 == 0 or g == 1):
                 # --- MODIFIED: Print all indicators for the current generation's best solution ---
                print(f"Gen {g} best   - Fitness: {current_fitness:.4f}, Cost: {current_cost:.2f}, v: {current_v}, r: {current_r}, u: {current_u}, s: {current_s}")
            
            self.generation_history.append({
                'generation': g, 
                'best_cost': self.best_cost, 
                'best_fitness': self.best_fitness,
                'best_v': self.best_v,
                'current_best_cost': current_cost, 
                'avg_cost': np.mean(cost_list)
            })
        
        end_time = time.time()
        self.ga_execution_time = end_time - start_time
        
        if verbose:
            print(f"\nGA completed after {self.max_gen} generations")
            print(f"Final best cost={self.best_cost:.2f} (Fitness={self.best_fitness:.4f})")
            print(f"Final best indicators - v: {self.best_v}, r: {self.best_r}, u: {self.best_u}, s: {self.best_s}")
            print(f"Execution time: {self.ga_execution_time:.2f} seconds")
        
        selected_columns = self.get_selected_columns()
        return 1.0, self.best_cost, selected_columns, None, None

    def log_ga_execution(self, experiment_log_df):
        """Log execution metrics to experiment dataframe."""
        metrics = ["ga_execution_time", "best_cost", "best_fitness", "best_v", "best_r", "best_u", "best_s"]
        log_ids = [f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}{i}" for i in range(len(metrics))]
        values = [self.ga_execution_time, self.best_cost, self.best_fitness, self.best_v, self.best_r, self.best_u, self.best_s]
        
        df = pd.DataFrame({
            "log_ids": log_ids,
            "experiment_name": [self.experiment_name] * len(metrics),
            "step": ["genetic_algorithm"] * len(metrics),
            "metric": metrics,
            "value": values,
        })

        df = pd.concat([experiment_log_df, df], ignore_index=True)
        return df

def run_ga(columns, timetables_path, experiment_name, experiment_log_df, tmax_list, ga_params=None, verbose=True, csv_results_path="ga_results.csv"):
    """
    Run GA with provided columns and log results to CSV.
    """
    ga = GeneticAlgorithm(columns, timetables_path, experiment_name, ga_params)
    best_fitness, best_cost, selected_columns, PSI, C_min = ga.run(verbose)
    ga_execution_log = ga.log_ga_execution(experiment_log_df)
    
    v, r, u, s = ga.best_v, ga.best_r, ga.best_u, ga.best_s
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"selected_columns_{timestamp}_{experiment_name}.txt"

    if tmax_list is not None:
        tmax_list = "-".join(list(map(str, tmax_list)))
    
    with open(filename, "w") as f:
        f.write(f"GA Results Summary - {timestamp}\n")
        f.write("="*50 + "\n")
        f.write(f"Repeated trips: {r}\n")
        f.write(f"Uncovered trips: {u}\n")
        f.write(f"Short routes: {s}\n")
        f.write(f"Number of vehicles: {v}\n")
        f.write(f"Best cost: {best_cost:.2f}\n")
        f.write(f"Best fitness: {ga.best_fitness:.4f}\n")
        f.write(f"Number of selected routes: {len(selected_columns)}\n")
        f.write("="*50 + "\n\n")

        if ga_params:
            f.write("GA Parameters:\n")
            f.write("-"*30 + "\n")
            for k, v_param in ga_params.items():
                f.write(f"{k}: {v_param}\n")
            f.write(f"tmax_list: {tmax_list}\n")
            f.write("\n")
        
        f.write("Selected Columns Details:\n")
        f.write("-"*30 + "\n")
        for k, v_col in selected_columns.items():
            f.write(f"{k}:\n")
            for _k, _v in v_col.items():
                f.write(f"\t{_k}: {_v}\n")
            f.write("\n")
    
    csv_row = {
        'tmax_list': tmax_list, 'ga_run_filename': filename, 'experiment_name': experiment_name,
        'repeated_trips': r, 'uncovered_trips': u, 'short_routes': s,
        'number_of_vehicles': v, 'best_cost': best_cost, 'best_fitness': ga.best_fitness, 
        'selected_routes_count': len(selected_columns)
    }
    
    default_params = {
        'max_gen': None, 'crossover_p': None, 'mutation_p': None, 'pop_size': None,
        'elite_indivs': None, 'gene_segments': None, 'L_min_constant': None,
        'L_max_constant': None
    }
    
    if ga_params:
        for p_name in default_params:
            csv_row[p_name] = ga_params.get(p_name, None)
    else:
        csv_row.update(default_params)
    
    new_row_df = pd.DataFrame([csv_row])
    
    header = not os.path.exists(csv_results_path)
    new_row_df.to_csv(csv_results_path, mode='a', header=header, index=False)
    
    if verbose:
        print(f"Results saved to: {filename}")
        print(f"CSV log updated: {csv_results_path}")
        print(f"Summary: {v} vehicles, {r} repeated trips, {u} uncovered trips, {s} short routes")
    
    return best_fitness, best_cost, selected_columns, ga_execution_log