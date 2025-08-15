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
    print("v:", v, "r:", r, "u:", u, "s:", s)
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
    
    fitness_list = []
    cost_list = []
    g = 1
    best_fit = math.inf
    best_cost = math.inf
    
    while g <= max_gen:
        
        print()
        print(f"GENERATION {g} --------------------------------------------------")
        print(f"Best fit: {best_fit} | Best cost: {best_cost}")
        
        if g == 1:
            population = initialize_population()
            for ind in population:
                eval_result = evaluate_individial(ind)
                fitness_list.append(eval_result[0])
                cost_list.append(eval_result[1])
            elite = np.argsort(fitness_list)[-E_num:]
            best = elite[-1]

            best_fit = fitness_list[best]
            best_cost = cost_list[best]

            g+=1

            print(sorted(fitness_list, reverse=True))
            # break
            continue

        new_population = []

        while len(new_population) < pop_size:
            parents = spin_roulette(fitness_list, population)

            u = np.random.uniform(0.0, 1.0)
            if u <= crossover_p:
                offspring_1, offspring_2 = make_crossover(parents, gene_segments)
            else:
                offspring_1 = parents[0]
                offspring_2 = parents[1]

            offspring_1 = mutate(offspring_1)
            offspring_2 = mutate(offspring_2)

            new_population.append(offspring_1)
            new_population.append(offspring_2)

        new_fitness_list = []
        new_cost_list = []
        for _i, new_ind in enumerate(new_population):
            new_eval_result = evaluate_individial(new_ind)
            new_fitness_list.append(new_eval_result[0])
            new_cost_list.append(new_eval_result[1])
        sorted_new_fitness = np.argsort(new_fitness_list)
        new_elite = sorted_new_fitness[-E_num:]
        new_best = new_elite[-1]
        new_worst = sorted_new_fitness[0]
        #
        if new_fitness_list[new_best] > fitness_list[best]:

            new_population.append(population[best])
            new_fitness_list.append(fitness_list[best])
            new_cost_list.append(cost_list[best])
            new_fitness_list.pop(new_worst)
            new_cost_list.pop(new_worst)
            new_population.pop(new_worst)
            fitness_list = new_fitness_list
            cost_list = new_cost_list
            population = new_population
            elite = np.argsort(fitness_list)[-E_num:]
            best = elite[-1]
            best_fit = fitness_list[best]
            best_cost = cost_list[best]
            g+=1

        else:
            temp_elite = [{"generation": g-1, "fitness": fitness_list[x], "index": x} for x in elite]
            temp_elite.extend(
                [{"generation": g, "fitness": new_fitness_list[x], "index": x} for x in new_elite]
            )
            temp_elite.sort(key=lambda x: x["fitness"], reverse=True)

            temp_elite = temp_elite[:E_num]

            temp_elite_fitness_sum = sum([x["fitness"] for x in temp_elite])
            
            for item in temp_elite:
                item['prop'] = item["fitness"] / temp_elite_fitness_sum

            best_id = np.random.choice(
                a=list(range(0, len(temp_elite))),
                size=1,
                p=[x["prop"] for x in temp_elite]
            )[0]
            
            best_data = temp_elite[best_id]

            if best_data['generation'] == g-1:
                new_population.append(population[best_data['index']])
                new_fitness_list.append(fitness_list[best_data['index']])
                new_cost_list.append(cost_list[best_data['index']])
                new_fitness_list.pop(new_worst)
                new_cost_list.pop(new_worst)
                new_population.pop(new_worst)
            else:
                new_population.append(new_population[best_data['index']])
                new_fitness_list.append(new_fitness_list[best_data['index']])
                new_cost_list.append(new_cost_list[best_data['index']])
                new_fitness_list.pop(new_worst)
                new_cost_list.pop(new_worst)
                new_population.pop(new_worst)

            population = new_population
            fitness_list = new_fitness_list
            cost_list = new_cost_list

            elite = np.argsort(fitness_list)[-E_num:]
            best = elite[-1]

            best_fit = fitness_list[best]
            best_cost = cost_list[best]

            # for i, ind in enumerate(population):
            #     print(i, ind, fitness_list[i], cost_list[i])
            #     print()
            # print("Elite indexes:", elite)
            # print("Best index:", best)
            # print("best_fit:", best_fit)
            # print("best_cost:", best_cost)

            # break
            print(sorted(fitness_list, reverse=True))
            break
            g+=1

                

run_ga()






        


        

            






#parent_1 = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
#parent_2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
#offspring_1, offspring_2 = crossover([parent_1, parent_2], 3)
#print(offspring_1)
#print(offspring_2)


        
        



    










# for chromo in population:
#     v = sum(chromo)
#     for i, gene in enumerate(chromo):
#         if gene == 0:
#             continue
#         else:
#             k = col_keys[i]
#             flat_trips.extend(columns[k]['Path'])
#             cost += columns[k]["Cost"]
#             data = columns[k]["Data"]
#             op_time = sum([float(data["total_dh_time"]), float(data["total_travel_time"]), float(data["total_wait_time"])])
#             operation_times.append(float(op_time))
# flat_trips = [item for item in flat_trips if item.startswith("l")]
# flat_trips_len = len(flat_trips)
# unique_trips = set(flat_trips)
# r = flat_trips_len - len(unique_trips)
# u = len(list(filter(lambda x: x not in timetable_trips["trip_id"].values, unique_trips)))
# s = sum([1 if item < 480 else 0 for item in operation_times])


# fx = w4 / ((w1 * v) + (w2 * (r + u)) + (w3 * s))

# print(fx)

    




