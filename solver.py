# -*- coding: utf-8 -*-
"""
Created on Tue May  1 21:03:12 2018

@author: admin
"""

import numpy as np

import functools

import datetime

import sys

def timer():
   now = datetime.timedelta.now()
   return now

def parse_stp_file(filename):
    f = open(filename)
    
    adj_list = []
    terminal_list = []
    node_count = 0
    
    current_zone = 0
    
    for line in f:
        splt = line.split(' ')
        if len(splt) > 0 and splt[0].startswith("END"):
            current_zone = 0
            continue
        elif len(splt) > 1:
            if splt[0].startswith("Nodes"):
                node_count = int(splt[1])
                continue
            elif splt[1].startswith("Graph"):
                current_zone = 1
                continue
            elif splt[1].startswith("Terminals"):
                current_zone = 2
                continue
        if current_zone == 1 and len(splt) == 4 and splt[0] == "E":
            adj_list.append(Edge(int(splt[1]), int(splt[2]), int(splt[3])))
            
        elif current_zone == 2 and splt[0] == "T":
            terminal_list.append(int(splt[1]))
            
    return adj_list, [i + 1 for i in range(node_count)], terminal_list
        
        #for tree_edge_idx in range(len(tree)):
         #   if tree[tree_edge_idx] == 1 and has_common_extremity(adj_list[]):
                

class Edge:
    def __init__(self, u, v, w):
        self.underlying = [u, v, w] 


    def has_common_extremity(self, e):
        return self.underlying[0] == e.underlying[0] \
            or self.underlying[0] == e.underlying[1] \
            or self.underlying[1] == e.underlying[0] \
            or self.underlying[1] == e.underlying[1]  
            
    def is_incident(self, v):
        return self.get_first() == v or self.get_second() == v
    
    def get_extremity(self, v):
        if self.get_first() == v:
            return 0
        elif self.get_second() == v:
            return 1
        
        return -1
            
    def get_weight(self):
        return self.underlying[2]
    
    def set_weight(self, w):
        self.underlying[2] = w
    
    def get_first(self):
        return self.underlying[0]
    
    def get_second(self):
        return self.underlying[1]
    
    def get(self, idx):
        return self.underlying[idx]
    
    def flip(self):
        return Edge(self.get_second(), self.get_first(), self.get_weight())
    
    def __repr__(self):
        return "(" + str(self.get_first()) + ", " + str(self.get_second()) + ") : " + str(self.get_weight())

# A simple Union-Find structure using the path compression and rank heuristic techniques
class UnionFind:
    def __init__(self):
        self.parent = self
        self.rank = 0
        
    def find(x):
        if not x.parent is x:
            x.parent = UnionFind.find(x.parent)
        return x.parent
    
    def union(x, y):
        xRoot = UnionFind.find(x)
        yRoot = UnionFind.find(y)
        
        if xRoot != yRoot:
            if xRoot.rank < yRoot.rank:
                xRoot.parent = yRoot
            else:
                yRoot.parent = xRoot
                if xRoot.rank == yRoot.rank:
                    xRoot.rank += 1

class Graph:
    def __init__(self, vertices, edge_list = []):
        self.V = list(vertices)
        self.E = list(edge_list)
        
    def insert(self, e):
        self.E.append(e)
   
    def get_edge(self, i):
        self.E[i];
    
    def get_edge_count(self):
        return len(self.E)
    
    def get_vertex_count(self):
        return len(self.V)
    
    def get_total_weight(self):
        s = 0
        
        for edge in self.E:
            s += edge.get_weight()
            
        return s
        
    def generate_mst(self):
        new_list = sorted(self.E, key=functools.cmp_to_key(lambda x1, x2: 1 if x1.get_weight() > x2.get_weight() else -1))

        union_sets = {}
        
        for vertice in self.V:
            union_sets[vertice] = UnionFind()
        
        tree = Graph(self.V, [])
                
        for edge in new_list:
            if tree.get_edge_count() == (tree.get_vertex_count() - 1):
                break
            
            v1 = edge.get_first()
            v2 = edge.get_second()
            
            if not (v1 in union_sets and v2 in union_sets):
                continue
            
            if UnionFind.find(union_sets[v1]) != UnionFind.find(union_sets[v2]):
                tree.insert(edge)
                UnionFind.union(union_sets[v1], union_sets[v2])
                
        return tree
        
    def get_graphviz_code(self):
        graph_str = ""
        
        for edge in self.E:
            graph_str += str(edge.get_first()) + "--" + str(edge.get_second()) + "[label=" + str(edge.get_weight()) + "];"
            
        return graph_str
            
        
    
class PopulationGenerator:
    def __init__(self, graph, terminals, proba_generator, threshold, jitter = 0.4):
        self.genetic_code_length = graph.get_vertex_count() - len(terminals)
        self.set_proba_generator(proba_generator)
        self.set_threshold(threshold)
        self.set_jitter(jitter)
        
    def set_proba_generator(self, gen):
        self.generator = gen
    
    def set_threshold(self, threshold):
        self.threshold = threshold
        
    def set_jitter(self, jitter):
        self.jitter = jitter
        
    def generate(self, size):
        pop = []
        half_jitter = self.jitter * 0.5
        
        for i in range(size):
            new_individual = [0 for _ in range(self.genetic_code_length)]
            for j in range(self.genetic_code_length):
                generated = self.generator()
                jittering = np.random.random() * self.jitter - half_jitter
                
                if generated <= (self.threshold + jittering):
                    new_individual[j] = 1
                  
            pop.append(new_individual)
            
        return pop
  
class HeuristicPopulationGenerator:
    def individual_from_graph(self, graph):
        individual = [0 for _ in range(len(self.v_list))]
        
        index_dict = {}
        
        for idx in range(len(self.v_list)):
            index_dict[self.v_list[idx]] = idx
        
        for e in graph.E:
            if e.get_first() in index_dict:
                individual[index_dict[e.get_first()]] = 1
            if e.get_second() in index_dict:
                individual[index_dict[e.get_second()]] = 1
                            
        return individual
    
class ShortestPathHeuristicPopulationGenerator(HeuristicPopulationGenerator):
    def __init__(self, graph, terminals, jitter = 0.4):
        self.graph = graph
        self.set_terminals(terminals)
        self.set_jitter(jitter)
        
    def set_orig_graph(self, graph):
        self.graph = graph
        self.update_v_list
        
    def set_terminals(self, terminals):
        self.terminals = terminals
        self.update_v_list()
        
    def update_v_list(self):
        self.v_list = set(self.graph.V).difference(set(self.terminals))
        self.v_list = list(self.v_list)
        
    def set_jitter(self, jitter):
        self.jitter = jitter
        
    def shortest_path_heuristic(self, graph):
        terminal_count = len(self.terminals)
        
        new_graph = Graph(graph.V)
        
        construction_dict = {}
        #print("h")
        for i in range(terminal_count - 1):
            #print(i)
            node_list = [v for v in self.terminals]
            node_list += self.v_list
            

            res_dict = {}
        
            for v in node_list:
                res_dict[v] = (-1, None, None)
                
            res_dict[node_list[i]] = (0, None, None)
            #print(res_dict)
            
            node_list_len = len(node_list)
            
            for _ in range(node_list_len):
                min_idx = -1
                current_min_weight = 0
                
                for idx in range(len(node_list)):
                    weight = res_dict[node_list[idx]][0]
                    #print(weight, current_min_weight)
                    #print(res_dict)
                    if (min_idx == -1 or current_min_weight > weight) and weight != -1:
                        min_idx = idx
                        current_min_weight = weight
                        
                chosen_vertex = node_list[min_idx]
                chosen_weight = current_min_weight
                                
                del node_list[min_idx]
                
                for edge in graph.E:
                    v = None
                    
                    #if (edge.get_first() == 39 or edge.get_second() == 39) and chosen_vertex == 11:
                        #print(chosen_vertex, edge)
                        #print("YAY !!!")
                    
                    if edge.get_first() in node_list and edge.get_second() == chosen_vertex:
                        v = edge.get_first()  
                    elif edge.get_second() in node_list and edge.get_first() == chosen_vertex:
                        v = edge.get_second()
                    else:
                        continue
                    
                    w = res_dict[v][0]
                    
                    if w == -1 or chosen_weight + edge.get_weight() < w:
                        #print(w, chosen_weight + edge.get_weight())
                        res_dict[v] = (chosen_weight + edge.get_weight(), chosen_vertex, edge)

            #print(res_dict)
            
            for v in [v for v in self.terminals]:
                if not v == self.terminals[i]:
                    new_graph.insert(Edge(self.terminals[i], v, res_dict[v][0]))
                
            construction_dict[self.terminals[i]] = res_dict
            
        new_graph = new_graph.generate_mst()
        res_graph = Graph(graph.V)
        #print(construction_dict[11])
        for edge in new_graph.E:
            starting_point = edge.get_first()
            current_point = edge.get_second()
            if not edge.get_second() in construction_dict[starting_point]:
                starting_point = edge.get_second()
                current_point = edge.get_first()
                
            d = construction_dict[starting_point]

            _, prev_v, prev_e = d[current_point]
            while not prev_v is None:
                res_graph.insert(prev_e)
                current_point = prev_v
                _, prev_v, prev_e = d[current_point]
                
        v_list = set(self.graph.V).difference(set(self.terminals))
        v_list = list(self.v_list)
        
        res_graph = res_graph.generate_mst()
        
        for v in v_list:
            degree = 0
            last_incident_edge = None
            for e in res_graph.E:
                if e.is_incident(v):
                    degree +=1
                    last_incident_edge = e
            if degree <= 1:
                if not last_incident_edge is None:
                    res_graph.E.remove(last_incident_edge)
                             
        return res_graph.generate_mst()
    
                    
    def shortest_spanning_tree_heuristic(self, graph):
        
        vertices = set(self.graph.V).difference(set(self.terminals))
        v_list = list(vertices)
                
        res_graph = Graph(graph.V, list(graph.E))
        res_graph = res_graph.generate_mst()
        
        need_continue = True
        while need_continue:
            need_continue = False
        
            for v in v_list:
                degree = 0
                last_incident_edge = None
                for e in res_graph.E:
                    if e.is_incident(v):
                        degree +=1
                        last_incident_edge = e
                if degree <= 1:
                    if not last_incident_edge is None:
                        res_graph.E.remove(last_incident_edge)
                        vertices.remove(v)
                        need_continue = True
                        
        return Graph(list(vertices) + self.terminals, res_graph.E)
    
    def generate(self, size):
        half_jitter = self.jitter / 2
        
        population = []

        for _ in range(size):
            new_edge_list = []
            for e in self.graph.E:
                jittering = np.random.random() * self.jitter - half_jitter
                new_e = Edge(e.get_first(), e.get_second(), (max(0, e.get_weight() + e.get_weight() * jittering)))
                new_edge_list.append(new_e)
                
            new_g1 = self.shortest_path_heuristic(Graph(self.graph.V, new_edge_list))
            population.append(self.individual_from_graph(new_g1))
        
        return population 
    
class ShortestSpanningTreeHeuristicPopulationGenerator(HeuristicPopulationGenerator):
    def __init__(self, graph, terminals, jitter = 0.4):
        self.graph = graph
        self.set_terminals(terminals)
        self.set_jitter(jitter)
        
    def set_orig_graph(self, graph):
        self.graph = graph
        self.update_v_list
        
    def set_terminals(self, terminals):
        self.terminals = terminals
        self.update_v_list()
        
    def update_v_list(self):
        self.v_list = set(self.graph.V).difference(set(self.terminals))
        self.v_list = list(self.v_list)
        
    def set_jitter(self, jitter):
        self.jitter = jitter
           
    def shortest_spanning_tree_heuristic(self, graph):
        
        vertices = set(self.graph.V).difference(set(self.terminals))
        v_list = list(vertices)
                
        res_graph = Graph(graph.V, list(graph.E))
        res_graph = res_graph.generate_mst()
        
        need_continue = True
        while need_continue:
            need_continue = False
        
            for v in v_list:
                degree = 0
                last_incident_edge = None
                for e in res_graph.E:
                    if e.is_incident(v):
                        degree +=1
                        last_incident_edge = e
                if degree <= 1:
                    if not last_incident_edge is None:
                        res_graph.E.remove(last_incident_edge)
                        vertices.remove(v)
                        need_continue = True
                        
        return Graph(list(vertices) + self.terminals, res_graph.E)
    
    def generate(self, size):
        half_jitter = self.jitter / 2
        
        population = []

        for _ in range(size):
            new_edge_list = []
            for e in self.graph.E:
                jittering = np.random.random() * self.jitter - half_jitter
                new_e = Edge(e.get_first(), e.get_second(), (max(0, e.get_weight() + e.get_weight() * jittering)))
                new_edge_list.append(new_e)
                
            new_g1 = self.shortest_spanning_tree_heuristic(Graph(self.graph.V, new_edge_list))
            population.append(self.individual_from_graph(new_g1))
        
        return population 

            
class MixedPopulationGenerator:
    def __init__(self, generator_list, proportions):
        self.generator_list = generator_list
        self.propotions = proportions
        
    def generate(self, size):
        res = []
        
        for i in range(len(self.generator_list)):
            res += self.generator_list[i].generate(int(size * self.propotions[i]))

        return res            
    
class GenericSolver:
    def __init__(self, graph, terminals, penality_factor = 1000):
        self.graph = graph
        self.set_terminals(terminals)
        self.set_penality_factor(penality_factor)

        
    def set_graph(self, graph):
        self.graph = graph
        self.update_v_list()
        
    def set_terminals(self, terminals):
        self.terminals = terminals
        self.update_v_list()
        
    def set_penality_factor(self, factor):
        self.penality_factor = factor
        
    def update_v_list(self):
        self.v_list = set(self.graph.V).difference(set(self.terminals))
        self.v_list = list(self.v_list)
        
        
    #def elitist_replacement(l):
    
    def graph_from_individual(self, individual):
        new_v_list = []
        
        for i in range(len(individual)):
            gene = individual[i]
            if gene == 1:
                new_v_list.append(self.v_list[i])
                                
        return Graph(new_v_list + self.terminals, self.graph.E)
    
    def compute_fitness(self, individual_graph):
        mst = individual_graph.generate_mst()
        
        approx_fitness = mst.get_total_weight()
        e_count = mst.get_edge_count()
        expected_v_count = mst.get_vertex_count() - 1
        #print("e : ", e_count, expected_v_count)

        if e_count == (expected_v_count):    
            return approx_fitness
        else:
            return approx_fitness + self.penality_factor * (expected_v_count - e_count)
        
 
class GeneticCrossoverOperators:
    def one_point_crossover(p1, p2):
        assert(len(p1) == len(p2))
        
        crossover_point = int(np.random.random() * len(p1) - 0.1)
        child1 = p1[:crossover_point] + p2[crossover_point:]
        child2 = p2[:crossover_point] + p1[crossover_point:]
                
        return child1, child2

    def two_points_crossover(p1, p2):
        assert(len(p1) == len(p2))
        
        crossover_point1 = int(np.random.random() * len(p1) - 0.1)
        crossover_point2 = int(np.random.random() * len(p1) - 0.1)
        
        if crossover_point1 > crossover_point2:
            crossover_point1, crossover_point2 = crossover_point2, crossover_point1
        
        child1 = p1[:crossover_point1] + p2[crossover_point1:crossover_point2] + p1[crossover_point2:]
        child2 = p2[:crossover_point1] + p1[crossover_point1:crossover_point2] + p2[crossover_point2:]

        return child1, child2
  
    
class GeneticSolver(GenericSolver):
    def __init__(self, graph, terminals, generator = None, pop_size = 100, mutation_factor = 0.01):
        super().__init__(graph, terminals)
        
        if generator is None:
            self.population_generator = PopulationGenerator(graph, terminals, np.random.random, 0.35)
        else:
            self.population_generator = generator
            
        self.set_population_size(pop_size)
        self.set_mutation_factor(mutation_factor)
        self.crossover_operator = GeneticCrossoverOperators.one_point_crossover
        
        
        #self.population_generator = HeuristicPopulationGenerator(graph, terminals)
        
    def set_population_size(self, size):
        self.population_size = size
        
    def set_mutation_factor(self, factor):
        self.mutation_factor = factor 
        
    def set_crossover_operator(self, crossover_operator):
        self.crossover_operator = crossover_operator
    
    def execute_roulette_trial(fitness_list, fitness_sum, worst_fitness):
        roulette_result = np.random.random() * fitness_sum
        #print(worst_fitness)
        acc = 0
        for i in fitness_list:
            acc += worst_fitness - i[1]
            #print(roulette_result, acc)
            if roulette_result <= acc:
                #print("here")
                return i[0]
        
    def total_replacement(self, l):
        l = sorted(l, key = lambda k : k[1])
        chosen_pop = []
        
        fitness_sum = 0
        worst_fitness = -1
        for i in l:
            worst_fitness = max(worst_fitness, i[1])
        for i in l:
            fitness_sum += worst_fitness - i[1]


        """for i in l:
            if np.random.random() * i[1] <= best_fitness:
                parent_pop.append(i[0])
                
        print(len(parent_pop))"""
        while len(chosen_pop) <= self.population_size:
                            
            p1 = GeneticSolver.execute_roulette_trial(l, fitness_sum, worst_fitness)
            p2 = GeneticSolver.execute_roulette_trial(l, fitness_sum, worst_fitness)
            child1, child2 = self.crossover_operator(p1, p2)
            
            m1 = np.random.random()
            m2 = np.random.random()
            
            if m1 <= self.mutation_factor:
                bit = int(np.random.random() * len(self.v_list)) - 1
                child1[bit] = 1 - child1[bit]
            if m2 <= self.mutation_factor:
                bit = int(np.random.random() * len(self.v_list)) - 1
                child2[bit] = 1 - child2[bit]
            
            chosen_pop.append(child1)
            chosen_pop.append(child2)
            
        return chosen_pop
    
    def elitist_replacement(self, l):
        chosen_pop = [(i, self.compute_fitness(self.graph_from_individual(i))) for i in self.total_replacement(l)]
        chosen_pop += l
        chosen_pop= sorted(chosen_pop, key = lambda k : k[1])
                
        return [i[0] for i in chosen_pop[:self.population_size]]
    
    def best_individual_from_pop(self, pop):
        best_individual = None
        best_fit = -1
        for i in range(len(pop)):
            g = self.graph_from_individual(pop[i])
            fitness = self.compute_fitness(g)
            if best_fit == -1 or fitness <= best_fit:
                best_individual = pop[i]
                best_fit = fitness
                
        return best_individual
            
    def solve(self, iter_count = 100, time_limit = 0):
        current_time = datetime.datetime.now()

        pop = self.population_generator.generate(self.population_size)
        current_pop_fitness = []
                
        initial_time = datetime.datetime.now()
        
        for _ in range(iter_count):
            #print(pop)
            if time_limit != 0 and (current_time - initial_time).total_seconds() >= time_limit:
                break 
            
            for i in pop:
                g = self.graph_from_individual(i)
                #print(self.compute_fitness(g))
                current_pop_fitness.append((i, self.compute_fitness(g)))
                
            
            pop = self.elitist_replacement(current_pop_fitness)
            #print("best fitness : ", current_pop_fitness)
            current_pop_fitness = []
            
            best_individual = None
            best_fit = -1
            for i in range(len(pop)):
                g = self.graph_from_individual(pop[i])
                fitness = self.compute_fitness(g)
                if best_fit == -1 or fitness <= best_fit:
                    best_individual = pop[i]
                    best_fit = fitness
                    
            current_time = datetime.datetime.now()
        
        best_individual = self.best_individual_from_pop(pop)         
                
        return self.graph_from_individual(best_individual), best_individual

class LocalSearchSolver(GenericSolver):
    def __init__(self, graph, terminals, generator = None, time_limit = 60):
        if generator is None:
            self.generator = ShortestPathHeuristicPopulationGenerator(graph, terminals)
        else :
            self.generator = generator
        self.time_limit = time_limit
        super().__init__(graph, terminals)
        
    def solve(self):
        current_time = datetime.datetime.now()
        initial_time = current_time
        
        current_best = None
        current_best_fitness = None
        
        fitness_list = []
        
        while (current_time - initial_time).total_seconds() <= self.time_limit:
            current = self.generator.generate(1)[0]
            g = self.graph_from_individual(current)
            #print(g.get_graphviz_code())
            #current = self.generator.individual_from_graph(g)
            
            current_fitness = self.compute_fitness(g)
            new_fitness = current_fitness + 1
    
            while not current_fitness == new_fitness and (current_time - initial_time).total_seconds() <= self.time_limit: 
                fitness_list.append(current_best_fitness)

                for i in range(len(current)):
                    new_individual = list(current)
                    new_individual[i] = 1 - new_individual[i]
                    g = self.graph_from_individual(new_individual)
                    self.set_penality_factor(-2 * g.generate_mst().get_total_weight())
                    fitness = self.compute_fitness(g)
                    
                    if fitness >= 0 and new_fitness > fitness:
                        current = new_individual
                        new_fitness = fitness
                current_fitness = new_fitness
                current_time = datetime.datetime.now()
                
            if current_best_fitness is None or current_best_fitness > new_fitness:
                current_best_fitness = new_fitness
                current_best = current
                           
        #print(fitness_list)             
        return self.graph_from_individual(current_best), current_best

if __name__ == '__main__':
        
    adj_list, nodes, term_list = parse_stp_file(sys.argv[1])
    
    g = Graph(nodes, adj_list)
    
    
    gen2 = ShortestSpanningTreeHeuristicPopulationGenerator(g, term_list, 0.6)
    gen1 = ShortestPathHeuristicPopulationGenerator(g, term_list, 0.6)
    gen3 = PopulationGenerator(g, term_list, np.random.random, 0.35)
    
    #solver = GeneticSolver(g, term_list, gen2)
    solver = GeneticSolver(g, term_list, MixedPopulationGenerator([gen1, gen2, gen3], (0.05, 0.015, 0.8)))
    #solver = GeneticSolver(g, term_list)
    #solver.set_mutation_factor(0.15)
    #solver.set_crossover_operator(GeneticCrossoverOperators.two_points_crossover)
    #solver.set_penality_factor(100)
    #solver = LocalSearchSolver(g, term_list, gen2)
    start = datetime.datetime.now()
    solved = solver.solve()
    end = datetime.datetime.now()
    print((end - start).total_seconds() * 1000)
    print(solved[0].generate_mst().get_total_weight())
    #print(solved[1])
    
    '''solver = GeneticSolver(g, term_list, 20)
    solver.set_mutation_factor(0.15)
    solved = solver.solve(1000)
    print(solved.generate_mst().get_total_weight())
    print(solved.generate_mst().E)
    print(solved.generate_mst().get_vertex_count())'''
    
    '''l = solved.V
    print(l)
    
    #solved.display_graph("testb13")
    for e in solved.generate_mst().E:
        print(e.get_first(), "--", e.get_second(), ";")
        
        if e.get_first() in l:
            print("remove : ", e.get_first())
            l.remove(e.get_first())
            
        if e.get_second() in l:
            print("remove : ", e.get_second())
            l.remove(e.get_second())
        print(l)
        
    print(l)'''


#b13 : [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
#b04 : [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0]
#b06 : [0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0]
# Unable to get to the optimum with heuristics
#b10 : [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
#b13 : [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
#c02 : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#c01 : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]