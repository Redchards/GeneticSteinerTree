# -*- coding: utf-8 -*-
"""
Created on Fri May 18 11:14:07 2018

@author: admin
"""

def install_and_import(package):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        import pip
        pip.main(['install', package])
    finally:
        globals()[package] = importlib.import_module(package)


install_and_import("numpy")


import urllib.request
import sys
from html.parser import HTMLParser
import os
import tarfile
import solver as SolverLib
from multiprocessing import Process, Manager


class InstanceParser(HTMLParser):
    in_instance = False
    in_data = False
    good_instance = False
    
    data_tag = "td"
    instance_tag = "tr"
    
    data_list = []
    
    def __init__(self, instance_list):
        self.instance_list = instance_list
        
        super().__init__()
    
    def handle_starttag(self, tag, attrs):
        if tag == self.data_tag:
            self.in_data = True
        if tag == self.instance_tag:
            self.in_instance = True
            
    def handle_data(self, data):
        if self.in_data:
            if not self.good_instance and self.in_instance:
                data = str(data)
                instance_data = data[2:]
                if instance_data[0] == "0":
                    instance_data = instance_data[1:]
                if len(self.instance_list) == 0 or str(instance_data) in self.instance_list:
                    self.good_instance = True
                    self.data_list.append([])
                else:
                    self.in_instance = False
            elif self.good_instance:
                if data.strip().isdigit():
                    self.data_list[-1].append(int(data))
                    
    def handle_endtag(self, tag):
        if tag == self.data_tag:
            self.in_data = False
        if tag == self.instance_tag:
            self.in_instance = False
            self.good_instance = False
            
 
def execute_solver(solver, res, solver_command = ()):
    solved = solver.solve(*solver_command)
    res.append(solver.compute_fitness(solved[0]))

if __name__ == '__main__':
    test_set = sys.argv[1]
    instances = sys.argv[2:]
    
    url_part = "http://steinlib.zib.de/showset.php?"
    down_url = "http://steinlib.zib.de/download/"
    
    if not os.path.isdir("./" + test_set):
        print(down_url + test_set + ".tgz")
        filename = test_set + ".tgz"
        fullfilename = os.path.join(os.path.dirname(os.path.realpath(__file__)),  filename)
        urllib.request.urlretrieve(down_url + filename, fullfilename)
        
        tar = tarfile.open(fullfilename, "r:gz")
        tar.extractall("sets")
        tar.close()
    print("opening ", url_part + test_set)
    with urllib.request.urlopen(url_part + test_set) as response:
        
        parser = InstanceParser(instances)
        parser.feed(str(response.read()))
        data_list = parser.data_list
        
        if len(instances) == 0:
            instances = [i for i in range(len(parser.data_list))]
        else:
            instances = [int(i) for i in instances]
    
    res_list = []    
    
    instance_tab_row = "Instances "
    
    algorithm_list =  ["PRGA", "SPGA", "STGA", "MGA1", "MGA2", "LSSP", "LSST"]
    
    for instance in instances:
        manager = Manager()
        
        ist = str(instance)
        if len(ist) == 1:
            ist = "0" + ist
        instance_name = test_set.lower() + ist
        instance_tab_row += "& " + instance_name
        res = manager.list([])
        res_list.append(res)

        adj_list, nodes, term_list = SolverLib.parse_stp_file(os.path.join("sets", test_set, instance_name + ".stp"))
        
        g = SolverLib.Graph(nodes, adj_list)
        
        gen1 = SolverLib.ShortestPathHeuristicPopulationGenerator(g, term_list, 0.6)
        gen2 = SolverLib.ShortestSpanningTreeHeuristicPopulationGenerator(g, term_list, 0.6)
        gen3 = SolverLib.PopulationGenerator(g, term_list, numpy.random.random, 0.35)
        
        genetic_solvers = [
            ("PRGA", SolverLib.GeneticSolver(g, term_list)),
            ("SPGA", SolverLib.GeneticSolver(g, term_list, gen1)),
            ("STGA", SolverLib.GeneticSolver(g, term_list, gen2)),
            ("MGA1",  SolverLib.GeneticSolver(g, term_list, SolverLib.MixedPopulationGenerator([gen1, gen2, gen3], (0.01, 0.019, 0.8)))),
            ("MGA2",  SolverLib.GeneticSolver(g, term_list, SolverLib.MixedPopulationGenerator([gen2, gen3], (0.02, 0.8))))]
    
    
        local_search_solvers = [
            ("LSSP", SolverLib.LocalSearchSolver(g, term_list, gen1, 300)),
            ("LSST", SolverLib.LocalSearchSolver(g, term_list, gen2, 300))]
        
        for name, solver in genetic_solvers: 
            print("* Benchmark " + name + " on " + instance_name)
            p = Process(target=execute_solver, args=(solver, res, (1000, 300)))
            p.start()
            p.join(400)
            if p.is_alive():
                print("ohno ... the code took too long :'(")
                p.terminate()
                p.join()
                res.append(-1)
        
        for name, solver in local_search_solvers:
            print("* Benchmark " + name + " on " + instance_name)
            p = Process(target=execute_solver, args=(solver, res))
            p.start()
            p.join(600)
            if p.is_alive():
                p.terminate()
                p.join()
                res.append(-1)
        
    instance_tab_row += "\\\\"
    # Now build the tab
    tab = "\\begin{table}[h]\n\t\centering\n\t\\begin{tabular}{"
    tab += "|l"
    
    for instance in instances:
        tab += "|c"
        
    tab += "|}\n"
    
    tab += instance_tab_row + "\n"
    
    efficiency_res = []
    
    for a in algorithm_list:
        efficiency_res.append((a, []))
    
    compute_efficiency = lambda found, opt: (found - opt) / opt
    
    for i in range(len(res_list)):
        inst_data = data_list[i]
        data_run = res_list[i]

        for j in range(len(efficiency_res)):
            if data_run[j] == -1:
                efficiency_res[j][1].append(None)
            else:
                efficiency_res[j][1].append(compute_efficiency(data_run[j], inst_data[-1]))        
    
    for algo_name, results in efficiency_res:
        tab += algo_name + " "
    
        for res in results:
            if res is None:
                tab +="& >5min"
            else:
                tab += "& " + "{0:.2f}".format(res) + "\%"
        
        tab += "\\\\ \n"
  
    tab += "\end{tabular}\n"
    tab += "\end{table}\n" 
    
    file = open(test_set + "_" + "_".join([str(i) for i in instances]) + "_benchmark", "w+")
    file.write(tab)