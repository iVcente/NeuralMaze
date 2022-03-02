from Individual import Individual
from Maze import Maze
from NeuralNetwork import NeuralNetwork
from Neuron import Neuron

from random import uniform
from typing import List
import argparse
import math
import random
import sys

def parse_maze(input_file: str) -> List[List[str]]:
    maze = []
    with open(input_file, 'r') as file:
        next(file)
        for line in file:
            row = "".join(line.split())
            maze.append(list(row))
    return maze

def manhattan_distance(current_pos: List[int], exit_pos: List[int]) -> int:
    # | (x2 - x1) | + | (y2 - y1) |
    delta_x = abs(exit_pos[0] - current_pos[0])
    delta_y = abs(exit_pos[1] - current_pos[1])
    
    return delta_x + delta_y

def activation_function(v: float) -> float:
    return math.tanh(v)

def model_network_with_individual(maze: Maze, individual: Individual, network: NeuralNetwork):
    individual.fitness = 0
    individual.gold = 0
    individual.path = []
    # Set network weights
    i = 0
    for neuron in network.hidden_layer:
        neuron.weights = individual.genes[i : i + neuron.n_weights]
        i += neuron.n_weights

    for neuron in network.output_layer:
        neuron.weights = individual.genes[i : i + neuron.n_weights]
        i += neuron.n_weights

    # Move player across the maze with network model
    while (True):
        # Input layer
        encoded_neighbors = maze.get_neighbors() # Free cell, gold, wall, exit, off limits
        distance = manhattan_distance(maze.current_pos, maze.exit_pos)
        encoded_neighbors.append(distance)
        # Insert values into hidden layer
        for neuron in network.hidden_layer:
            neuron.input_values = encoded_neighbors
        # Process hidden layer values | Generate V values
        v_values = [] # Size 4 -> 4 neurons in hidden layer
        for neuron in network.hidden_layer:
            value = 0
            w = neuron.weights
            value = w[0] + (encoded_neighbors[0] * w[1]) + (encoded_neighbors[1] * w[2]) + \
                    (encoded_neighbors[2] * w[3]) + (encoded_neighbors[3] * w[4]) + \
                    (encoded_neighbors[4] * w[5])
            v_values.append(value)
        # Process hidden layer values | Generate Y values
        y_values = []
        for v in v_values:
            y = activation_function(v)
            y_values.append(y)
        # Insert Y values into output layer
        for neuron in network.output_layer:
            neuron.input_values = y_values
        # Process output layer | Generate V values
        v_values = [] # Size 4 -> 4 neurons in output layer    
        for neuron in network.output_layer:
            w = neuron.weights
            value = w[0] + (encoded_neighbors[0] * w[1]) + (encoded_neighbors[1] * w[2]) + \
                    (encoded_neighbors[2] * w[3]) + (encoded_neighbors[3] * w[4])
            v_values.append(value)
        # Process output layer values | Generate Y values
        y_values = []
        for v in v_values:
            y = activation_function(v)
            y_values.append(y)

        # Insert Y value into output layer y_value
        i = 0
        for neuron in network.output_layer:
            neuron.y_value = y_values[i]
            i += 1

        # Get output layer move
        max_neuron = network.output_layer[0]
        for neuron in network.output_layer:
            if (neuron.y_value > max_neuron.y_value):
                max_neuron.y_value = neuron.y_value
                max_neuron.role = neuron.role

        next_direction = max_neuron.role
        individual.path.append(next_direction)

        # Moves player in maze
        if (maze.previous_pos == maze.check_next_pos(next_direction)):
            individual.fitness += -25
            break

        element = maze.traverse(next_direction)
        score = maze.get_element_score(element)
        individual.fitness += score

        if (element == "Gold"):
            individual.gold += score

        if (element == "Wall" or element == "Off limits"):
            break

        if (element == "Exit"):
            print('\n')
            print(f"{'-' * 100 : ^130}")
            print('\n')
            print(f"{'|========== Path to exit found! ==========|' : ^130}\n")
            print(f"{'-> Genes: '} {individual.genes}")
            print(f"\n{'-> Fitness: '} {individual.fitness}")
            print(f"{'-> Gold: '} {individual.gold}")
            print(f"{'-> Path: '} {individual.path}")
            sys.exit(0)

    maze.reset_position()

def fitness(maze: Maze, population: List[Individual], network: NeuralNetwork):
    for individual in population:
        model_network_with_individual(maze, individual, network)

def elitism(population: List[Individual]) -> Individual:
    return max(population, key = lambda individual: individual.fitness)

def crossover(population: List[Individual], crossover_rate: float) -> List[Individual]:
    new_population = []
    
    for i in range(len(population) - 1): # -1 because we already picked one individual with elitism
        # Tournament
        # Pick 2 random individuals and do a tournament to get the most fit
        father = elitism(random.choices(population, k = 2))
        mother = elitism(random.choices(population, k = 2))

        if (random.random() < crossover_rate): # Generates a number between 0.0 and 1.0, then compares with crossover rate
            child_genes = []

            for gene in range(len(mother.genes)):
                child_gene = (father.genes[gene] + mother.genes[gene]) / 2.0
                child_genes.append(child_gene)

            new_population += [Individual(child_genes)]
        else:
            new_population += [elitism([father, mother])] # Get most fit between father and mother

    return new_population

def mutation(new_population: List[Individual], weights_range: List[float], mutation_rate: float) -> List[Individual]:    
    mutated_population = []
    for individual in new_population:
        new_genes = []
        for gene in individual.genes:
            if (random.random() < mutation_rate): # Generates a number between 0.0 and 1.0, then compares with mutation rate
                mutation = uniform(weights_range[0], weights_range[1])
                new_genes.append(mutation)
            else:
                new_genes.append(gene)
        mutated_population.append(Individual(new_genes))

    return mutated_population

def main() -> None:
    # Parsing arguments
    parser = argparse.ArgumentParser(description = "Neural maze \o/")
    parser.add_argument("-f","--file", help = "Maze file", type = str, required = True)
    parser.add_argument("-p","--population", help = "Initial population", type = int, required = True)
    parser.add_argument("-g","--generations", help = "Max number of generations", type = int, required = True)
    parser.add_argument("-c","--crossover", help = "Crossover rate", type = float, required = True)
    parser.add_argument("-m","--mutation", help = "Mutation rate", type = float, required = True)
    parser.add_argument("-e","--execution", help = "Execution mode: fast or slow", type = str, required = True)
    args = parser.parse_args()

    # Get maze and info about it
    maze = Maze(parse_maze(args.file))
    possible_directions = ['N', 'S', 'W', "E"]    

    # Create neural network
    hidden_layer_neuron_n_weights = 6
    hidden_layer_size = 4
    hidden_layer = []
    output_layer_neuron_n_weights = 5
    output_layer_size = 4
    output_layer = []
    n_weights = (hidden_layer_neuron_n_weights * hidden_layer_size) + (output_layer_neuron_n_weights * output_layer_size)

    for i in range(hidden_layer_size):
        hidden_layer.append(Neuron(hidden_layer_neuron_n_weights, "Hidden"))

    for i in range(output_layer_size):
        output_layer.append(Neuron(output_layer_neuron_n_weights, possible_directions[i]))

    neural_network = NeuralNetwork(hidden_layer, output_layer)

    # Generate initial population | first generation | generation 0
    weights_range = [-1.0, 1.0]
    population = [] # List of individuals/chromosomes

    for i in range(args.population):
        genes = []
        for gene in range(n_weights): # There are a total of 44 weights on the neural network
            genes.append(uniform(weights_range[0], weights_range[1])) # Weights range
        individual = Individual(genes)
        population.append(individual)

    # Start evolution
    most_fit = population[0]
    for generation in range(args.generations):
        ### Fitness calculation | Heuristic function (Neural network)
        fitness(maze, population, neural_network)

        if (args.execution == "fast"):
            current_most_fit = elitism(population)
            print(f"\n|====== Population generation: {generation} =====|\n")
            print("-> Generation's most fit individual: ")
            print(f"\t-> Genes: {current_most_fit.genes}")
            print(f"\n\t-> Fitness: {current_most_fit.fitness}")
            print(f"\t-> Gold: {current_most_fit.gold}")
            print(f"\t-> Path: {current_most_fit.path}")
        elif (args.execution == "slow"):
            # Print population
            print(f"\n|====== Population generation: {generation} =====|\n")
            for individual in range(len(population)):
                print(f"Generation: {generation} | Individual {individual} ---> Fitness: {population[individual].fitness}")
                print(f"-> Gold: {population[individual].gold}")
                print(f"Path: {population[individual].path}")
        else:
            print("Please, choose \"fast\" or \"slow\" for execution mode.")

        ### Selection and Reproduction
        current_most_fit = elitism(population)
        new_population = crossover(population, args.crossover)
        mutated_population = mutation(new_population, weights_range, args.mutation)
        population = mutated_population
        population.append(most_fit)

        if (current_most_fit.fitness > most_fit.fitness):
            most_fit = current_most_fit

    print('\n')
    print(f"{'-' * 100 : ^130}")
    print('\n')
    print(f"{'|========== Solution not found. Best solution up to this moment ==========|' : ^130}\n")
    print(f"{'-> Genes:'} {most_fit.genes}")
    print(f"\n{'-> Fitness:'} {most_fit.fitness}")
    print(f"{'-> Gold:'} {most_fit.gold}")
    print(f"{'-> Path:'} {most_fit.path}")

if (__name__ == "__main__"):
    main()

# Recommended population size: ~100, ~250 or ~500 | Default: 500
# Recommended number of generations: ~100, ~250, ~500 or ~1000
# Recommended crossover rate: ~0.5
# Recommended mutation rate: ~0.05
# https://www.youtube.com/watch?v=MacVqujSXWE
