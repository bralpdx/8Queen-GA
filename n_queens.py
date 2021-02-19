"""
Implementation of N-Queens problem
with the use of a Genetic Algorithm.

The chromosome variable in __generate_population
is structured such that the index in chromosome[index] represents the column,
and the value at chromosome[index] represents the row.

Solutions are given in index form. I.e. [0, 1, 2, 3, ...] = rows 1, 2, 3, 4, ...
"""
from random import randint, uniform
from matplotlib import pyplot as plt
import numpy as np
import board
import sys

N = 8  # Size of the game-board
pop_size = 100  # The size of the population
MUTPCT = 8  # Mutation percentage
# The maximum number of non-attacking pairs
MAX_FIT = int(np.math.factorial(N) / (np.math.factorial(2) * np.math.factorial(N - 2)))
MAX_ITER = 700  # Maximum iterations (generations)


class NQueens:
    def __init__(self):
        self.population = list()
        self.__generate_population()

    # Generates a starting population
    def __generate_population(self):
        for i in range(pop_size):
            chromosome = list()
            for col in range(N):
                chromosome.append(randint(0, 7))
            self.population.append(chromosome)

    # Returns the current number of attacking pieces
    def number_of_attacks(self, chromosome):
        attacks = 0
        try:
            attacks += sum([chromosome.count(q) - 1 for q in chromosome])/2  # Represents the row attacks
        except TypeError:
            print("Population Error: ", chromosome)

        for i in range(N):
            for j in range(i, N):
                if i != j:
                    if abs(i - j) == abs(chromosome[i] - chromosome[j]):
                        attacks += 1
        return attacks

    # Returns the fitness value of an individual chromosome
    def fitness(self, chromosome):
        return int(MAX_FIT - self.number_of_attacks(chromosome))

    # Returns a list of selection probability values for parents
    def survival_prob(self):
        prob_list = list()
        for i in range(pop_size):
            fit_list = [self.fitness(x) for x in self.population]  # list of fitness scores
            prob_list.append(self.fitness(self.population[i]) / np.sum(fit_list))
        return prob_list

    def breed(self):
        new_population = list()
        s_probs = self.survival_prob()  # list of parent survival probabilities
        pop_with_probs = zip(self.population, s_probs)  # tuples of chromosomes and corresponding probability
        sum_of_probs = sum(prob for chrom, prob, in pop_with_probs)  # the sum of all probabilities

        for i in range(pop_size):
            p1 = self.select_parent(self.population, s_probs, sum_of_probs)
            p2 = self.select_parent(self.population, s_probs, sum_of_probs)

            j = 0
            # ensures that different parents are chosen
            while p1 == p2 and j < pop_size:
                p2 = self.population[j]
                j += 1

            child = self.crossover(p1, p2)
            new_population.append(child)

            if self.solution(child):
                return child

        return new_population

    def select_parent(self, pop, probs, sum, rand=False):
        r_value = uniform(0, sum)  # selects random value between 0 and the sum of all probabilities
        current = 0

        # if rand is true, return a random chromosome to be a parent
        if rand:
            index = randint(0, N-1)
            return self.population[index]

        # Iterates through all chromosomes until random value is met or exceeded
        for chrom, prob, in zip(pop, probs):
            if current + prob >= r_value:
                return chrom
            current += prob

    def crossover(self, p1, p2):
        cp = randint(1, N - 1)  # selects random crossover point in p1
        child = p1[:cp] + p2[cp:]

        # MUTPCT% chance of mutation
        if randint(1, 100) <= MUTPCT:
            self.__mutate(child)
        return child

    def __mutate(self, child):
        mp = randint(0, N - 1)  # selects random index to mutate
        child[mp] = randint(0, N - 1)  # sets the queen at the index to a random row

    def solution(self, chromosome):
        if self.fitness(chromosome) == MAX_FIT:
            return True
        return False


# Calculates the average fitness of each generation
def avg_fit(fit_vals, gen):
    sum = 0
    for i in fit_vals:
        sum += i
    avg = sum / pop_size
    return [gen, avg]


# Generates a plot ofAverage fitness over generations
def plot(y_vals):
    x = np.array([])
    y = np.array([])

    for i in y_vals:
        x = np.append(x, i[0])
        y = np.append(y, i[1])

    plt.title("Average Fitness Per Generation")
    plt.xlabel("Generations")
    plt.ylabel("Avg Fitness")
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    game = NQueens()
    generations = 1
    fit_vals = list()
    avg_fitness_list = list()
    solved = False

    while not solved and generations < MAX_ITER:
        for i in range(len(game.population)):
            fit_vals.append(game.fitness(game.population[i]))
            solved = game.solution((game.population[i]))
        avg_fitness_list.append(avg_fit(fit_vals, generations))
        fit_vals = []
        game.population = game.breed()
        if len(game.population) == N:  # Breaks if a solution is found
            break
        generations += 1

    plot(avg_fitness_list)  # generates a graph of average fitness over generation

    if generations == MAX_ITER:
        print("Max generations reached. No solution found.")
        print("[Generation, Average Fitness]")
        for i in avg_fitness_list:
            print(i)
        sys.exit(1)

    print("generations: ", generations)
    print("Solution: ", game.population)
    print("[Generation, Average Fitness]")
    for i in avg_fitness_list:
        print(i)

    board = board.Board(game.population)
