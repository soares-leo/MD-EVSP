import pandas as pd
import numpy as np
import math
import time
import datetime
import json

class GeneticAlgorithm:
    """
    Genetic Algorithm for column selection optimization.
    Accepts columns directly as input parameter instead of loading from file.
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
        self.w1 = params.get('w1', 0.6)
        self.w2 = params.get('w2', 2)
        self.w3 = params.get('w3', 0.4)
        self.w4 = params.get('w4', 1000)
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
        
        # Cache for fitness evaluations
        self.FITNESS_CACHE = {}
        self.total_cache_evaluations = 0
        
        # Results storage
        self.best_individual = None
        self.best_fitness = -math.inf
        self.best_cost = math.inf
        self.generation_history = []

    def initialize_population(self):
        """Initialize population with valid individuals."""
        initial_population = []
        print(f"Initializing population of size {self.pop_size}...")

        L_median = (self.L_min + self.L_max) / 2
        
        while len(initial_population) < self.pop_size:
            individual = (np.random.uniform(0.0, 1.0, len(self.columns)) >= 1 - (L_median / len(self.columns))).astype(np.int8)
            vehicles = individual.sum()
            
            if self.L_min <= vehicles <= self.L_max:
                initial_population.append(individual)
                
        return initial_population

    def evaluate_individual(self, individual):
        """
        Evaluate fitness of an individual.
        Uses caching to avoid redundant calculations.
        Returns: (fitness, cost, r, u, s, flat_trips_tuple)
        """
        individual_tuple = tuple(individual)
        if individual_tuple in self.FITNESS_CACHE:
            return self.FITNESS_CACHE[individual_tuple]

        v = individual.sum()  # Number of vehicles
        cost = 0
        operation_times = []
        flat_trips = []
        
        # Collect data from selected columns
        for i, gene in enumerate(individual):
            if gene == 1:
                k = self.col_keys[i]
                column_data = self.columns[k]
                flat_trips.extend(column_data['Path'])
                cost += column_data["Cost"]
                data = column_data["Data"]
                op_time = sum([
                    float(data["total_dh_time"]), 
                    float(data["total_travel_time"]), 
                    float(data["total_wait_time"])
                ])
                operation_times.append(float(op_time))

        # Filter for actual trips (starting with 'l')
        flat_trips = [item for item in flat_trips if item.startswith("l")]
        unique_trips = set(flat_trips)
        
        r = len(flat_trips) - len(unique_trips)  # Redundant trips
        u = self.total_trips - len(unique_trips)  # Uncovered trips
        s = sum(1 for item in operation_times if item < 480)  # Short routes
        
        # Calculate fitness
        denominator = (self.w1 * v) + (self.w2 * (r + u)) + (self.w3 * s)
        fitness = math.e ** (0.1 * (self.w4 / denominator if denominator != 0 else math.inf))
        
        # Cache result - always store the full tuple
        flat_trips_tuple = tuple(flat_trips)
        result = (fitness, cost, r, u, s, flat_trips_tuple)
        self.FITNESS_CACHE[individual_tuple] = result
        
        return result

    def spin_roulette(self, fitness_list, population):
        """Roulette wheel selection for parent selection."""
        fitness_array = np.array(fitness_list)
        total_fitness = fitness_array.sum()
        
        if total_fitness == 0:
            probs = None
        else:
            probs = fitness_array / total_fitness
            
        parent_indices = np.random.choice(
            a=len(population), 
            size=2, 
            p=probs, 
            replace=False
        )
        
        return population[parent_indices[0]], population[parent_indices[1]]

    def make_crossover(self, parents):
        """Perform crossover between two parents."""
        parent_1, parent_2 = parents
        offspring_1 = []
        offspring_2 = []
        
        num_genes = len(parent_1)
        
        for i in range(0, num_genes, self.gene_segments):
            p1_segment = parent_1[i:i + self.gene_segments]
            p2_segment = parent_2[i:i + self.gene_segments]
            
            if np.random.uniform(0.0, 1.0) <= 0.5:
                offspring_1.extend(p2_segment)
                offspring_2.extend(p1_segment)
            else:
                offspring_1.extend(p1_segment)
                offspring_2.extend(p2_segment)
                
        return np.array(offspring_1, dtype=np.int8), np.array(offspring_2, dtype=np.int8)

    def mutate(self, individual):
        """Apply mutation to an individual."""
        mutation_mask = np.random.uniform(0.0, 1.0, len(individual)) <= self.mutation_p
        individual[mutation_mask] = 1 - individual[mutation_mask]
        return individual

    def get_selected_columns(self):
        """
        Extract the columns selected by the best individual.
        Returns a dictionary with sequential route numbering.
        """
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
        """
        Run the genetic algorithm.
        
        Args:
            verbose: If True, print progress information
            
        Returns:
            tuple: (best_fitness, best_cost, selected_columns_dict)
        """
        g = 1
        elite_population = []
        elite_fitness = []
        elite_costs = []

        start_time = time.time()
        
        while g <= self.max_gen:
            if verbose and g % 100 == 1:  # Print every 100 generations
                print(f"\nGENERATION {g} " + "-"*50)
                print(self.best_fitness, self.best_cost)
                self.FITNESS_CACHE = {}
            
            if g == 1:
                # Initialize first generation
                population = self.initialize_population()
                fitness_list = []
                cost_list = []
                
                for ind in population:
                    eval_result = self.evaluate_individual(ind)
                    fitness_list.append(eval_result[0])  # fitness
                    cost_list.append(eval_result[1])     # cost
                
                population = np.array(population, dtype=np.int8)
                
                # Select elite
                elite_indices = np.argsort(fitness_list)[-self.elite_indivs:]
                elite_population = [population[i] for i in elite_indices]
                elite_fitness = [fitness_list[i] for i in elite_indices]
                elite_costs = [cost_list[i] for i in elite_indices]
                
                # Track best
                best_idx = elite_indices[-1]
                self.best_fitness = fitness_list[best_idx]
                self.best_cost = cost_list[best_idx]
                self.best_individual = population[best_idx].copy()
                               
                if verbose:
                    print(f"Initial best: fitness={self.best_fitness:.4f}, cost={self.best_cost:.2f}")
                
                g += 1
                continue
            
            # Evolution step
            previous_best_idx = np.argmax(fitness_list)
            previous_best_individual = population[previous_best_idx]
            previous_best_fitness = fitness_list[previous_best_idx]
            
            # Create new population
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
            
            # Evaluate new population
            new_fitness_list = []
            new_cost_list = []
            
            for ind in new_population:
                eval_result = self.evaluate_individual(ind)
                new_fitness_list.append(eval_result[0])  # fitness
                new_cost_list.append(eval_result[1])     # cost
            
            # Update elite pool
            new_elite_indices = np.argsort(new_fitness_list)[-self.elite_indivs:]
            new_elite_population = [new_population[i] for i in new_elite_indices]
            new_elite_fitness = [new_fitness_list[i] for i in new_elite_indices]
            new_elite_costs = [new_cost_list[i] for i in new_elite_indices]
            
            # Merge old and new elite
            temp_elite_pop = elite_population + new_elite_population
            temp_elite_fit = elite_fitness + new_elite_fitness
            temp_elite_cost = elite_costs + new_elite_costs
            
            master_elite_indices = np.argsort(temp_elite_fit)[-self.elite_indivs:]
            elite_population = [temp_elite_pop[i] for i in master_elite_indices]
            elite_fitness = [temp_elite_fit[i] for i in master_elite_indices]
            elite_costs = [temp_elite_cost[i] for i in master_elite_indices]
            
            # Elitism strategy
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
                    chosen_elite_idx = np.random.choice(len(elite_population), p=elite_probs)
                    
                    new_population[worst_new_idx] = elite_population[chosen_elite_idx]
                    new_fitness_list[worst_new_idx] = elite_fitness[chosen_elite_idx]
                    new_cost_list[worst_new_idx] = elite_costs[chosen_elite_idx]
            
            # Update population
            population = new_population
            fitness_list = new_fitness_list
            cost_list = new_cost_list
            
            # Track best overall
            current_best_idx = np.argmax(fitness_list)
            if fitness_list[current_best_idx] > self.best_fitness:
                self.best_fitness = fitness_list[current_best_idx]
                self.best_cost = cost_list[current_best_idx]
                self.best_individual = population[current_best_idx].copy()
                
                if verbose and g % 100 == 0:
                    print(f"New best: fitness={self.best_fitness:.4f}, cost={self.best_cost:.2f}")
            
            # Store history
            self.generation_history.append({
                'generation': g,
                'best_fitness': self.best_fitness,
                'best_cost': self.best_cost,
                'avg_fitness': np.mean(fitness_list),
                'cache_size': len(self.FITNESS_CACHE)
            })

            self.total_cache_evaluations += len(self.FITNESS_CACHE)
           
            g += 1
        
        if verbose:
            print(f"\nGA completed after {self.max_gen} generations")
            print(f"Final best: fitness={self.best_fitness:.4f}, cost={self.best_cost:.2f}")
            print(f"Cache hits saved {self.total_cache_evaluations} evaluations")
        
        # Return results with selected columns
        selected_columns = self.get_selected_columns()
        
        end_time = time.time()
        self.ga_execution_time = end_time - start_time
        
        return self.best_fitness, self.best_cost, selected_columns

    def log_ga_execution(self, experiment_log_df):
        metrics = ["ga_execution_time", "best_fitness", "best_cost"]
        log_ids = [f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}{i}" for i in range(len(metrics))]
        values = [self.ga_execution_time, self.best_fitness, self.best_cost]
        df = pd.DataFrame({
            "log_ids": log_ids,
            "experiment_name": [self.experiment_name] * len(metrics),
            "step": ["genetic_algorithm"] * len(metrics),
            "metric": metrics,
            "value": values,
        })

        df = pd.concat([experiment_log_df, df], ignore_index=True)
        return df


# Convenience function for backward compatibility
def run_ga(columns, timetables_path, experiment_name, experiment_log_df, ga_params=None, verbose=True):
    """
    Run GA with provided columns.
    
    Args:
        columns: Dictionary of columns to optimize
        timetables_path: Path to timetables CSV
        experiment_name: Name of the experiment
        experiment_log_df: DataFrame for logging
        ga_params: Optional GA parameters dictionary
        verbose: Print progress
    Returns:
        tuple: (best_fitness, best_cost, selected_columns, experiment_log_df)
    """
    ga = GeneticAlgorithm(columns, timetables_path, experiment_name, ga_params)
    best_fitness, best_cost, selected_columns = ga.run(verbose)
    ga_execution_log = ga.log_ga_execution(experiment_log_df)
    
    # Get detailed evaluation of the best individual
    best_individual_tuple = tuple(ga.best_individual)
    if best_individual_tuple in ga.FITNESS_CACHE:
        fitness, cost, r, u, s, flat_trips_tuple = ga.FITNESS_CACHE[best_individual_tuple]
    else:
        # Re-evaluate if not in cache (shouldn't happen, but safety check)
        eval_result = ga.evaluate_individual(ga.best_individual)
        fitness, cost, r, u, s, flat_trips_tuple = eval_result
    
    # Create timestamped filename
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"selected_columns_{timestamp}_{experiment_name}.txt"
    
    with open(filename, "w") as f:
        # Write summary information first
        f.write(f"GA Results Summary - {timestamp}\n")
        f.write("="*50 + "\n")
        f.write(f"Repeated trips: {r}\n")
        f.write(f"Uncovered trips: {u}\n")
        f.write(f"Short routes: {s}\n")
        f.write(f"Best fitness: {best_fitness:.6f}\n")
        f.write(f"Best cost: {best_cost:.2f}\n")
        f.write(f"Number of selected routes: {len(selected_columns)}\n")
        f.write("="*50 + "\n\n")

        f.write("GA Params:\n")
        f.write("-"*30 + "\n")
        for k, v in ga_params.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")
        
        # Write detailed column information
        f.write("Selected Columns Details:\n")
        f.write("-"*30 + "\n")
        for k, v in selected_columns.items():
            f.write(f"{k}:\n")
            for _k, _v in v.items():
                f.write(f"\t{_k}: {_v}\n")
            f.write("\n")
    
    if verbose:
        print(f"Results saved to: {filename}")
        print(f"Summary: {r} repeated trips, {u} uncovered trips, {s} short routes")
    
    return best_fitness, best_cost, selected_columns, ga_execution_log