import random
import numpy as np

# MYCIEL 3 için

path = 'datasets/myciel3.txt'
population_size = 50

# veri seti yükleme

def _load_graph(path):
    edges = []
    v = []
    vertices = []

    with open(path, mode='r') as f:
        for line in f.readlines():
            if line.split()[0] == 'e':
                line_ = line.lstrip('e')
                w1, w2 = line_.split()

                if w1 != w2:
                    edges.append([int(w1), int(w2)])
                    _, w = line_.split()
                    v.append(1)
                    v.append(int(w))
                    vertices = list(set(v))

        edges = np.array(edges, dtype=int)
        vertices = np.array(vertices, dtype=int)

    # print(edges)
    # print(vertices)
    return edges, vertices

# population
def init_population(num_nodes, max_color):
    # random.seed()

    population = np.zeros((population_size, num_nodes), dtype=int)
    for i in range(population_size):
        individual = []
        for j in range(num_nodes):
            color = random.randint(1, max_color)
            individual.append(color)
        population[i] = np.array(individual)
        individual = []
    # print(population)
    return population


# fitness / node
def nfitness(individual):
    fitness_value = 0
    dif_colors = []
    for i in range(len(individual) - 1):
        if individual[i] == individual[i + 1]:
            fitness_value += 0
        else:
            dif_colors.append(individual[i])
            dif_colors.append(individual[i + 1])
            fitness_value = len(set(dif_colors))

    #print(fitness_value)
    return fitness_value


def penalty(edges, individual):
    penalty = 0
    a = 1
    for i in edges:
        # node değiştiğinde;
        if a != i[0]:
            a += 1
        # veri setindeki her nodeun 1 kenarına bakıyoruz
        if a == i[0]:

            if individual[a - 1] == individual[i[1] - 1]:
                penalty += 1
                #print("node a: ", a)
                #print(individual[a - 1])
                #print("node i: ", i[1])
                #print(individual[i[1] - 1])
                #print("penalti artar")
                #print("------\n")
            else:
                penalty += 0

    #print("individual: ", individual)
    #print("penalty: ", penalty)
    #print("\n")
    return penalty


def tourney(edges, p1, p2, population):
    penalty1 = penalty(edges, population[p1])
    penalty2 = penalty(edges, population[p2])

    Parent = None

    if penalty1 == 0 and penalty2 == 0:
        fitness_p1 = nfitness(population[p1])
        fitness_p2 = nfitness(population[p2])
        minFitness = min(fitness_p2, fitness_p1)
        if fitness_p1 == minFitness:
            Parent = population[p1]
        else:
            Parent = population[p2]
    elif penalty1 != 0 and penalty2 != 0:
        minPenalty = min(penalty1, penalty2)
        if penalty1 == minPenalty:
            Parent = population[p1]
        else:
            Parent = population[p2]
    elif (penalty1 == 0 and penalty2 != 0) or (penalty1 != 0 and penalty2 == 0):
        if penalty1 == 0:
            Parent = population[p1]
        elif penalty2 == 0:
            Parent = population[p2]

    return Parent


def parent_selection(edges, population):
    size = len(population) - 1
    p1 = random.randint(0, size)
    p2 = random.randint(0, size)
    p3 = random.randint(0, size)
    p4 = random.randint(0, size)
    if p1 == p2:
        p2 = random.randint(0, size)

    if p3 == p4:
        p3 = random.randint(0, size)

    Parent1 = tourney(edges, p1, p2, population)
    Parent2 = tourney(edges, p3, p4, population)

    #print("\n")
    #print("p1: ", Parent1)
    #print("p2: ", Parent2)

    return Parent1, Parent2


def crossover(Parent1, Parent2):
    crossover_point = random.randint(1, len(Parent1) - 1)
    #print("cross_over point: ", (crossover_point) + 1)
    child1 = np.concatenate((Parent1[:crossover_point], Parent2[crossover_point:]))
    child2 = np.concatenate((Parent2[:crossover_point], Parent1[crossover_point:]))
    #print("child1: ", child1)
    #print("child2: ", child2)
    return child1, child2


def mutation(individual, color):
    # Her gen için, random.random() fonksiyonu kullanılarak 0 ile 1 arasında bir rasgele olasılık değeri üretilir.
    # Bu olasılık değeri, mutation_rate'den küçükse, gen mutasyona uğrar.
    mutation_rate = 0.3
    color_size = len(color)

    mutated_individual = individual.copy()
    colorcpy = color.copy()

    for i in range(len(individual)):
        if random.random() < mutation_rate:
            mutatedGen = i
            mGen = mutatedGen + 1
            #print("mutatedGen:", mGen)
            randColor = random.randint(1, color_size)
            #print("ind_mutated:", mutated_individual[mutatedGen])
            #print("randColor: ", randColor)
            if mutated_individual[mutatedGen] != randColor:
                mutated_individual[mutatedGen] = randColor
            else:
                available_colors = [c for c in colorcpy if c != mutated_individual[mutatedGen]]
                randColor = random.choice(available_colors)
                mutated_individual[mutatedGen] = randColor

    # print(color)

    #print(individual)
    #print(mutated_individual)
    return mutated_individual


def new_generation(edges, population):
    new_pop = []
    for i in range(int(len(population) / 2)):
        Parent1, Parent2 = parent_selection(edges, population)
        Child1, Child2 = crossover(Parent1, Parent2)
        c1 = mutation(Child1, color)
        c2 = mutation(Child2, color)
        new_pop.append(c1)
        new_pop.append(c2)

    new_pop = np.array(new_pop)
    return new_pop


def survivor_selection(edges, population, new_generation):
    new_gen = []
    fitnessList = []
    penaltyList = []
    new_pop = []
    sorted_values = []
    sorted_List = []

    sayac = 0
    for i in population:
        fitness = nfitness(i)
        penaltyy = penalty(edges, i)
        fitnessList.append(fitness)
        penaltyList.append(penaltyy)
        #new_fit = (fitness * 0.4) + (penaltyy * 0.6)
        #new_fitList.append(new_fit)
        sayac = sayac + 1

    sayac2 = population_size
    for j in new_generation:
        fitness2 = nfitness(j)
        penaltyy2 = penalty(edges, j)
        fitnessList.append(fitness2)
        penaltyList.append(penaltyy2)
        #new_fit2 = (fitness2 * 0.4) + (penaltyy2 * 0.6)
        #new_fitList.append(new_fit2)
        sayac2 = sayac2 + 1

    max_fitness = max(fitnessList)
    min_fitness = min(fitnessList)
    max_penalty = max(penaltyList)
    min_penalty = min(penaltyList)

    index = 0
    for x,y in zip(fitnessList,penaltyList):
        x = (x - min_fitness) / (max_fitness - min_fitness)
        x = round(x, 2)
        fitnessList[index] = x
        y = (y - min_penalty) / (max_penalty - min_penalty)
        y = round(y, 2)
        penaltyList[index] = y
        z = (x * 0.4) + (y * 0.6)
        z = round(z , 2)
        new_pop.append([index,x,y,z])
        index = index + 1

    new_pop = np.array(new_pop)
    #print(new_pop)

    sorted_indices = np.argsort(new_pop[:, 3])  # z değerine göre sıralanmış indisler
    sorted_pop = new_pop[sorted_indices]  # new_pop'u sıralı şekilde depolar

    for item in sorted_pop:
        index, x, y, z = item
        #print(f"Index: {index}, x: {x}, y: {y}, z: {z}")
        sorted_values.append([index, x, y, z])

    sorted_List = np.array(sorted_values[:population_size])
    #print(sorted_List)

    for k in sorted_List[:, 0]:
        k = int(k)
        if k < population_size:
            new_gen.append(population[k])
        else:
            index = k - population_size
            new_gen.append(new_generation[index])

    new_gen = np.array(new_gen)
    #print("**\n")
    #print("YENI POPULASYON")
    #print(new_gen)
    #print("\nBirey Sayısı: ", len(new_gen))

    return new_gen



def dehb_survivor(edges, population, new_generation):
    fitList = []
    fitList2 = []
    zero_penalty = []
    sorted_values2 = []
    sorted_values3 = []
    dehb_gen = []
    new_gen_dehb = []

    sayac = 0
    for i in population:
        fitness = nfitness(i)
        penaltyy = penalty(edges, i)
        fitList.append([sayac, fitness, penaltyy])
        if penaltyy == 0:      #penaltısı 0 olanlar ayrı bir listeye atıldı.
            zero_penalty.append([sayac, fitness, penaltyy])
        else:
            fitList2.append([sayac, fitness, penaltyy])
        sayac = sayac + 1

    sayac2 = population_size
    for j in new_generation:
        fitness2 = nfitness(j)
        penaltyy2 = penalty(edges, j)
        fitList.append([sayac2, fitness2, penaltyy2])
        if penaltyy2 == 0:
            zero_penalty.append([sayac2, fitness2, penaltyy2])
        else:
            fitList2.append([sayac2, fitness2, penaltyy2])
        sayac2 = sayac2 + 1

    #penaltısı 0 olanlar sıralandı.
    size = population_size - len(zero_penalty)
    zero_penalty = np.array(zero_penalty)
    fitList2 = np.array(fitList2)


    sorted_zero = np.argsort(zero_penalty[:, 1])
    print(sorted_zero)
    sorted_pop_zero = zero_penalty[sorted_zero]

    for item2 in sorted_pop_zero:
        index, x, y = item2
        sorted_values2.append([index, x, y])


    #penaltısı 0 olmayanlar, penaltıya göre küçükten büyüğe sıralandı.
    sorted_fitList = np.argsort(fitList2[:, 2])
    sorted_fit = fitList2[sorted_fitList]

    for item3 in sorted_fit:
        index, x, y = item3
        sorted_values3.append([index, x, y])

    sorted_size = np.array(sorted_values3[:size])

    dehb_gen = np.concatenate((sorted_values2, sorted_size))
    #print(dehb_gen)
    #print(len(dehb_gen))

    #seçilen bireyler yeni listeye atıldı.
    for k in dehb_gen[:, 0]:
        k = int(k)
        if k < population_size:
            new_gen_dehb.append(population[k])
        else:
            index = k - population_size
            new_gen_dehb.append(new_generation[index])

    sorted_values2 = np.array(sorted_values2)
    sorted_values3 = np.array(sorted_values3)
    fitList = np.array(fitList)


    new_gen_dehb = np.array(new_gen_dehb)
    #print(new_gen_dehb)
    #print(len(new_gen_dehb))

    return new_gen_dehb



if __name__ == '__main__':
    max_iter = 200
    iter = 0
    _load_graph(path)
    vertices = _load_graph(path)[1]
    edges = _load_graph(path)[0]
    color = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    color_size = len(color)
    population = init_population(len(vertices), color_size)
    best_individual = []
    for iter in range(max_iter):
        child = new_generation(edges, population)
        new_gen = dehb_survivor(edges, population, child)
        best_fit = nfitness(new_gen[0])
        best_penalty = penalty(edges, new_gen[0])
        best_ind = new_gen[0]
        print(f"Iteration: {iter},Individual: {best_ind} ,fitness: {best_fit}, penalty: {best_penalty}")
        best_individual.append(best_ind)
        if best_penalty == 0 and best_fit == 4:
            #print(f"Iteration: {iter},Individual: {best_ind} ,fitness: {best_fit}, penalty: {best_penalty}")
            break

    best_individual = np.array(best_individual)
    #print(best_individual)

    sorted_best = np.empty((0, 3))

    _sayac = 0
    zero_list = np.empty((0, 3))
    for i in best_individual:
        fitness = nfitness(i)
        penalty_value = penalty(edges, i)
        if penalty_value == 0:
            inner_zero = np.array([_sayac, fitness, penalty_value])
            zero_list = np.vstack([zero_list, inner_zero])
        else:
            inner_array = np.array([_sayac, fitness, penalty_value])
            sorted_best = np.vstack([sorted_best, inner_array])

        _sayac = _sayac + 1

    sorted_values = np.empty((0,3))
    if len(zero_list) > 0:
        # penaltısı 0 olanlar sıralandı.
        sorted_indices_zero = np.argsort(zero_list[:, 1])
        sorted_pop_zero = zero_list[sorted_indices_zero]

        for item in sorted_pop_zero:
            index, x, y = item
            inner_sort = np.array([index, x, y])
            sorted_values = np.vstack([sorted_values, inner_sort])

    #print("zero list: ",sorted_values)

    sorted_indices = np.argsort(sorted_best[:, 2])
    sorted_pop = sorted_best[sorted_indices]

    if len(zero_list) > 0:
        zero_best = sorted_values[:, 0]
        a = int(zero_best[0])
        print("\nBEST ITERATION: ",a)
        print("BEST INDIVIDUAL: ", best_individual[a])
        print("BEST FITNESS: ", nfitness(best_individual[a]))
        print("BEST PENALTY: ", penalty(edges, best_individual[a]))
    else:
        bestind = sorted_pop[:, 0]
        b = int(bestind[0])
        print("\nBEST ITERATION: ",b)
        print("BEST INDIVIDUAL: ", best_individual[b])
        print("BEST FITNESS: ", nfitness(best_individual[b]))
        print("BEST PENALTY: ", penalty(edges, best_individual[b]))


