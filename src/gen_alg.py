"""!
@brief Simple implementation of the genetic algorithm. All functions borrowed from https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/
"""
from numpy.random import randint
from numpy.random import rand


def selection(pop, scores, k=3):
    '''!
    Makes a tournament selection from the population

    @param pop    initial population of random bitstring
    @param scores    the scores for each member of pop
    @param k    optional, number of selections made from the population

    @return a mutated list of bits
    '''
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k-1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]



def crossover(p1, p2, r_cross):
    '''!
    Implements a crossover between parents according to probability r_cross

    @param p1  one of the parent lists of bits
    @param p2  one of the parent lists of bits
    @param r_cross   probability that a crossover will occurr

    @return a mutated list of bits
    '''
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = randint(1, len(p1)-2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]



def mutation(bitstring, r_mut):
    '''!
    Flips bits according to the r_mut probability

    @param bitstring  the list of bits
    @param r_mut    probability that a mutation will occurr

    @return a mutated list of bits
    '''
    for i in range(len(bitstring)):
        # check for a mutation
        if rand() < r_mut:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]



def genetic_algorithm(objective, E, leaves, n_bits, n_iter, n_pop, r_cross, r_mut):
    '''!
    Simple implementation of the genetic algorithm assuming input is a list of bits

    @param objective   callable objective function. Takes a list of bits where each set of 3 is binary for a single integer value between 0 and 7, the param E, and the param leaves.
    @param E   An Nx6 matrix where each row corresponds to a line (as defined by two 3d points) representing a leaf
    @param leaves   a list where each element is a numpy array containing xyz points for each leaf
    @param n_bits   the number of bits in a single candidate solution
    @param n_iter   the number of desired algorithm iterations
    @param n_pop    the population size
    @param r_cross   probability that a crossover will occurr
    @param r_mut    probability that a mutation will occurr

    @return a list containing the best solution and it's cost
    '''
    # initial population of random bitstring
    pop = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]
    # keep track of best solution
    best, best_eval = 0, objective(pop[0], E, leaves)
    # enumerate generations
    for gen in range(n_iter):
        # evaluate all candidates in the population
        scores = [objective(c, E, leaves) for c in pop]
        # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                # print(">%d, new best f(%s) = %.3f" % (gen, pop[i], scores[i]))
        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c, r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children
    return [best, best_eval]