import pandas as pd
import numpy as np
import math
import time
import datetime
import json
import os

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
        
        # Phi values for each objective (as per paper, 0 < phi < 1)
        self.phi_v = params.get('phi_v', 0.01)
        self.phi_r = params.get('phi_r', 0.01)
        self.phi_u = params.get('phi_u', 0.8)
        self.phi_ru = params.get('phi_ru', 0.05)
        self.phi_s = params.get('phi_s', 0.2)
        
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
        
        # Results storage - track by COST, not fitness
        self.best_individual = None
        self.best_cost = math.inf
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
        
        while len(initial_population) < self.pop_size:
            individual = (np.random.uniform(0.0, 1.0, len(self.columns)) >= 1 - (L_median / len(self.columns))).astype(np.int8)
            vehicles = individual.sum()
            
            if self.L_min <= vehicles <= self.L_max:
                initial_population.append(individual)
                
        return initial_population
    
    def calculate_indicators(self, individual):
        """
        Calculate indicators and other useful data for further evaluation.
        Returns: (v, r, u, s, cost, flat_trips)
        """
        v = individual.sum()  # Number of vehicles
        cost = 0
        operation_times = []
        flat_trips = []
        
        # Collect data from selected columns
        for i, gene in enumerate(individual):
            if gene == 1:
                k = self.col_keys[i]
                column_data = self.columns[k]
                path_elements = column_data['Path']
                trips_only = [item for item in path_elements if item.startswith("l")]
                flat_trips.extend(trips_only)
                cost += column_data["Cost"]
                data = column_data["Data"]
                op_time = sum([
                    float(data["total_dh_time"]), 
                    float(data["total_travel_time"]), 
                    float(data["total_wait_time"])
                ])
                operation_times.append(float(op_time))

        unique_trips = set(flat_trips)
        
        r = len(flat_trips) - len(unique_trips)  # Redundant trips
        u = self.total_trips - len(unique_trips)  # Uncovered trips
        s = sum(1 for item in operation_times if item < 480)  # Short routes
        if self.use_objective:
            cost += u * 1000
        else:
            cost = (r + u + s) *3 / v

        return v, r, u, s, cost, flat_trips
    
    def calculate_fitness_parameters(self, v_list, r_list, u_list, s_list):
        """
        Calculate PSI parameters for fitness function based on current generation.
        Following Lima et al. (1996) dynamic selectivity scaling.
        """
        # Current generation statistics
        min_v = np.min(v_list)
        avg_v = np.mean(v_list)
        min_r = np.min(r_list)
        avg_r = np.mean(r_list)
        min_u = np.min(u_list)
        avg_u = np.mean(u_list)
        #min_ru = np.min(ru_list)
        #avg_ru = np.mean(ru_list)  
        min_s = np.min(s_list)
        avg_s = np.mean(s_list)
        
        C_min = np.array([min_u, min_r, min_s])
        C_avg = np.array([avg_u, avg_r, avg_s])
        
        # Calculate differences
        C_diff = C_avg - C_min
        
        # Calculate PSI using phi values (ln(phi) / (c_avg - c_min))
        # As per paper equation (25)
        phi_list = [self.phi_v, self.phi_r, self.phi_u, self.phi_s]
        PSI = np.array([
            math.log(phi) / c_diff if c_diff > 0 else 0 
            for phi, c_diff in zip(phi_list, C_diff)
        ])
        
        return PSI, C_min, C_avg
    
    def evaluate_fitness(self, individual, PSI, C_min):
        """
        Calculate fitness using dynamic selectivity scaling function.
        Based on Lima et al. (1996) equation (36) for multi-objective.
        
        Returns: fitness value
        """
        v, r, u, s, cost, _ = self.calculate_indicators(individual)
        ru = r + u
        
        # Current cost vector
        C = np.array([u, r, s])
        
        # Calculate fitness as per paper: f(c) = e^(PSI * (C - C_min))
        # For multi-objective, it's the product (or sum in log space)
        exponent = np.sum(PSI * (C - C_min))
        fitness = math.exp(exponent)
        
        return fitness

    def evaluate_population(self, population):
        """
        Evaluate entire population, calculating all metrics and fitness.
        Returns: (fitness_list, cost_list, v_list, r_list, u_list, s_list)
        """
        v_list, r_list, u_list, ru_list, s_list, cost_list = [], [], [], [], [], []
        
        # First pass: calculate indicators for all individuals
        for ind in population:
            v, r, u, s, cost, _ = self.calculate_indicators(ind)
            v_list.append(v)
            r_list.append(r)
            u_list.append(u)
            ru_list.append(r + u)
            s_list.append(s)
            cost_list.append(cost)
        
        # Calculate fitness parameters based on current generation
        PSI, C_min, C_avg = self.calculate_fitness_parameters(v_list, r_list, u_list, s_list)
        
        # Second pass: calculate fitness
        fitness_list = []
        for ind in population:
            fitness = self.evaluate_fitness(ind, PSI, C_min)
            fitness_list.append(fitness)
        
        return fitness_list, cost_list, v_list, r_list, u_list, s_list

    def spin_roulette(self, fitness_list, population):
        """
        Roulette wheel selection for parent selection.
        Uses fitness for probabilistic selection.
        """
        fitness_array = np.array(fitness_list)
        total_fitness = fitness_array.sum()
        
        if total_fitness == 0 or np.any(np.isnan(fitness_array)):
            # Fallback to uniform selection if fitness calculation fails
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

    def update_best_if_better(self, individual, cost):
        """
        Update best individual if the given one is better (lower cost).
        """
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_individual = individual.copy()
            
            # Recalculate all indicators for the new best
            v, r, u, s, _, _ = self.calculate_indicators(individual)
            self.best_v = v
            self.best_r = r
            self.best_u = u
            self.best_ru = r + u
            self.best_s = s
            return True
        return False

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
            tuple: (best_fitness, best_cost, selected_columns_dict, PSI, C_min)
        """
        start_time = time.time()
        
        # For tracking current generation best (for display)
        last_v, last_r, last_u, last_s, last_cost = 0, 0, 0, 0, 0
        
        # Elite tracking
        elite_population = []
        elite_costs = []
        elite_fitness = []
        
        population = None
        
        for g in range(1, self.max_gen + 1):
            
            if verbose and g % 100 == 1:
                print(f"\nGENERATION {g} " + "-"*50)
                print(f"Best overall - v: {self.best_v}, r: {self.best_r}, u: {self.best_u}, s: {self.best_s}, cost: {self.best_cost:.4f}")
                print(f"Last gen best - v: {last_v}, r: {last_r}, u: {last_u}, s: {last_s}, cost: {last_cost:.4f}")
            
            if g == 1:
                # Initialize first generation
                population = self.initialize_population()
                population = np.array(population, dtype=np.int8)
                
                # Evaluate initial population
                fitness_list, cost_list, v_list, r_list, u_list, s_list = self.evaluate_population(population)
                
                # Find and track the best individual (by COST)
                best_idx = np.argmin(cost_list)
                self.update_best_if_better(population[best_idx], cost_list[best_idx])
                
                # Initialize elite (by COST)
                elite_indices = np.argsort(cost_list)[:self.elite_indivs]
                elite_population = [population[i].copy() for i in elite_indices]
                elite_costs = [cost_list[i] for i in elite_indices]
                
                if verbose:
                    print(f"Initial best: cost={self.best_cost:.2f}, v={self.best_v}, r={self.best_r}, u={self.best_u}, s={self.best_s}")
                
            else:
                # Evolution step
                
                # Create new population
                new_population = np.empty_like(population)
                
                for i in range(0, self.pop_size, 2):
                    # Select parents using fitness-based roulette wheel
                    parent_1, parent_2 = self.spin_roulette(fitness_list, population)
                    
                    # Crossover
                    if np.random.uniform(0.0, 1.0) <= self.crossover_p:
                        offspring_1, offspring_2 = self.make_crossover((parent_1, parent_2))
                    else:
                        offspring_1, offspring_2 = parent_1.copy(), parent_2.copy()
                    
                    # Mutation
                    new_population[i] = self.mutate(offspring_1)
                    if i + 1 < self.pop_size:
                        new_population[i+1] = self.mutate(offspring_2)
                
                # Evaluate new population
                fitness_list, cost_list, v_list, r_list, u_list, s_list = self.evaluate_population(new_population)
                
                # Elite replacement: replace worst individual with best elite
                worst_idx = np.argmax(cost_list)
                if elite_costs:
                    best_elite_idx = np.argmin(elite_costs)
                    new_population[worst_idx] = elite_population[best_elite_idx].copy()
                    cost_list[worst_idx] = elite_costs[best_elite_idx]
                    # Re-evaluate the replaced individual's metrics
                    v, r, u, s, _, _ = self.calculate_indicators(new_population[worst_idx])
                    v_list[worst_idx] = v
                    r_list[worst_idx] = r
                    u_list[worst_idx] = u
                    s_list[worst_idx] = s
                
                # Update elite pool (combine old and new, keep best by cost)
                combined_pop = list(elite_population) + list(new_population)
                combined_costs = elite_costs + cost_list
                combined_fitness = elite_fitness + fitness_list
                elite_indices = np.argsort(combined_costs)[:self.elite_indivs]
                elite_population = [combined_pop[i].copy() for i in elite_indices]
                elite_costs = [combined_costs[i] for i in elite_indices]
                elite_fitness = [combined_fitness[i] for i in elite_indices]
                
                # Update population
                population = new_population
            
            # Track current generation best (for display)
            current_best_idx = np.argmin(cost_list)
            last_v = v_list[current_best_idx]
            last_r = r_list[current_best_idx]
            last_u = u_list[current_best_idx]
            last_s = s_list[current_best_idx]
            last_cost = cost_list[current_best_idx]
            
            # Update global best if better found
            if self.update_best_if_better(population[current_best_idx], cost_list[current_best_idx]):
                if verbose and g % 100 == 0:
                    print(f">>> New best found! Cost: {self.best_cost:.2f}")
            
            # Store history
            self.generation_history.append({
                'generation': g,
                'best_cost': self.best_cost,
                'best_v': self.best_v,
                'best_r': self.best_r,
                'best_u': self.best_u,
                'best_s': self.best_s,
                'current_best_cost': last_cost,
                'avg_cost': np.mean(cost_list)
            })
        
        end_time = time.time()
        self.ga_execution_time = end_time - start_time
        
        if verbose:
            print(f"\nGA completed after {self.max_gen} generations")
            print(f"Final best: cost={self.best_cost:.2f}")
            print(f"Final best indicators - v: {self.best_v}, r: {self.best_r}, u: {self.best_u}, s: {self.best_s}")
            print(f"Execution time: {self.ga_execution_time:.2f} seconds")
        
        # Return results
        selected_columns = self.get_selected_columns()
        
        # Return dummy values for PSI and C_min for compatibility
        return 1.0, self.best_cost, selected_columns, None, None

    def log_ga_execution(self, experiment_log_df):
        """Log execution metrics to experiment dataframe."""
        metrics = ["ga_execution_time", "best_cost", "best_v", "best_r", "best_u", "best_s"]
        log_ids = [f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}{i}" for i in range(len(metrics))]
        values = [self.ga_execution_time, self.best_cost, self.best_v, self.best_r, self.best_u, self.best_s]
        
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
def run_ga(columns, timetables_path, experiment_name, experiment_log_df, tmax_list, ga_params=None, verbose=True, csv_results_path="ga_results.csv"):
    """
    Run GA with provided columns and log results to CSV.
    
    Args:
        columns: Dictionary of columns to optimize
        timetables_path: Path to timetables CSV
        experiment_name: Name of the experiment
        experiment_log_df: DataFrame for logging
        ga_params: Optional GA parameters dictionary
        verbose: Print progress
        csv_results_path: Path to CSV file for logging results (default: "ga_results.csv")
    Returns:
        tuple: (best_fitness, best_cost, selected_columns, experiment_log_df)
    """
    ga = GeneticAlgorithm(columns, timetables_path, experiment_name, ga_params)
    best_fitness, best_cost, selected_columns, PSI, C_min = ga.run(verbose)
    ga_execution_log = ga.log_ga_execution(experiment_log_df)
    
    # Get final metrics
    v, r, u, s, _, _ = ga.calculate_indicators(ga.best_individual)
    
    # Create timestamped filename for detailed results
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"selected_columns_{timestamp}_{experiment_name}.txt"

    if tmax_list is not None:
        tmax_list = "-".join(list(map(lambda x: str(x),tmax_list)))
    
    with open(filename, "w") as f:
        # Write summary information
        f.write(f"GA Results Summary - {timestamp}\n")
        f.write("="*50 + "\n")
        f.write(f"Repeated trips: {r}\n")
        f.write(f"Uncovered trips: {u}\n")
        f.write(f"Short routes: {s}\n")
        f.write(f"Number of vehicles: {v}\n")
        f.write(f"Best cost: {best_cost:.2f}\n")
        f.write(f"Number of selected routes: {len(selected_columns)}\n")
        f.write("="*50 + "\n\n")

        if ga_params:
            f.write("GA Parameters:\n")
            f.write("-"*30 + "\n")
            for k, v_param in ga_params.items():
                f.write(f"{k}: {v_param}\n")
            f.write(f"tmax_list: {tmax_list}\n")
            f.write("\n")
        
        # Write detailed column information
        f.write("Selected Columns Details:\n")
        f.write("-"*30 + "\n")
        for k, v_col in selected_columns.items():
            f.write(f"{k}:\n")
            for _k, _v in v_col.items():
                f.write(f"\t{_k}: {_v}\n")
            f.write("\n")
    
    # Prepare data for CSV logging
    csv_row = {
        'tmax_list': tmax_list,
        'ga_run_filename': filename,
        'experiment_name': experiment_name,
        'repeated_trips': r,
        'uncovered_trips': u,
        'short_routes': s,
        'number_of_vehicles': v,
        'best_cost': best_cost,
        'selected_routes_count': len(selected_columns)
    }
    
    # Add GA parameters to the row (with default values if not provided)
    if ga_params:
        csv_row['max_gen'] = ga_params.get('max_gen', None)
        csv_row['crossover_p'] = ga_params.get('crossover_p', None)
        csv_row['mutation_p'] = ga_params.get('mutation_p', None)
        csv_row['pop_size'] = ga_params.get('pop_size', None)
        csv_row['elite_indivs'] = ga_params.get('elite_indivs', None)
        csv_row['gene_segments'] = ga_params.get('gene_segments', None)
        csv_row['L_min_constant'] = ga_params.get('L_min_constant', None)
        csv_row['L_max_constant'] = ga_params.get('L_max_constant', None)
        csv_row['phi_v'] = ga_params.get('phi_v', None)
        csv_row['phi_r'] = ga_params.get('phi_r', None)
        csv_row['phi_u'] = ga_params.get('phi_u', None)
        csv_row['phi_ru'] = ga_params.get('phi_ru', None)
        csv_row['phi_s'] = ga_params.get('phi_s', None)
    else:
        # Set default None values if ga_params is not provided
        csv_row.update({
            'tmax_list': tmax_list,
            'max_gen': None,
            'crossover_p': None,
            'mutation_p': None,
            'pop_size': None,
            'elite_indivs': None,
            'gene_segments': None,
            'L_min_constant': None,
            'L_max_constant': None,
            'phi_v': None,
            'phi_r': None,
            'phi_u': None,
            'phi_ru': None,
            'phi_s': None
        })
    
    # Create DataFrame from the row
    new_row_df = pd.DataFrame([csv_row])
    
    # Append to CSV file (create if it doesn't exist)
    if os.path.exists(csv_results_path):
        # File exists, append without header
        new_row_df.to_csv(csv_results_path, mode='a', header=False, index=False)
    else:
        # File doesn't exist, create with header
        new_row_df.to_csv(csv_results_path, mode='w', header=True, index=False)
    
    if verbose:
        print(f"Results saved to: {filename}")
        print(f"CSV log updated: {csv_results_path}")
        print(f"Summary: {v} vehicles, {r} repeated trips, {u} uncovered trips, {s} short routes")
    
    return best_fitness, best_cost, selected_columns, ga_execution_log