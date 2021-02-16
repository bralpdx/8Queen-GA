"""
Implementation of N-Queens problem
with the use of a Genetic Algorithm.

The chromosome variable in __generate_population
is structured such that the index in chromosome[index] represents the column,
and the value at chromosome[index] represents the row.

"""
from random import randint
import numpy as np

N = 8  # Size of the game-board
pop_size = 10  # The size of the population
# The maximum number of non-attacking pairs
MAX_FIT = int(np.math.factorial(N) / (np.math.factorial(2) * np.math.factorial(N - 2)))


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
        attacks = abs(len(chromosome) - len(np.unique(chromosome)))  # Represents the row and column attacks
        for i in range(N):
            for j in range(i, N):
                if i != j:
                    if abs(i - j) == abs(chromosome[i] - chromosome[j]):
                        attacks += 1
        return attacks

    # Returns the fitness value of an individual chromosome
    def fitness(self, chromosome):
        return MAX_FIT - self.number_of_attacks(chromosome)

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
        for i in range(pop_size):
            p1 = []
            p2 = []
            child = self.crossover(p1, p2)

    def crossover(self, p1, p2):
        pass

    def mutate(self):
        pass


if __name__ == "__main__":
    game = NQueens()
    generations = 1
    fitness_scores = list()
    """
    while 0 not in fitness_scores:
        for i in range(len(game.population)):
            fitness_scores.append(game.fitness(game.population[i]))
        generations += 1
    """
    print(game.population)
    game.survival_prob()
    game.breed()
