# The beginning of the rewrite of responsenet

import numpy as np
from ortools.graph.python import min_cost_flow
from ortools.linear_solver import pywraplp
import argparse
import networkx as nx
import math

def parse_nodes(node_file):
    ''' Parse a list of sources or targets and return a set '''
    with open(node_file) as node_f:
        lines = node_f.readlines()
        nodes = set(map(str.strip, lines))
    return nodes

def construct_digraph(edges_file):
    """
    Similar to MinCostFlow, we need to parse a list of undirected edges and returns a graph object and idDict
    
    Parameters:
        edges_file : PATH()
            the PATH file for an interactome
    
    Returns:
        @G: graph object
        @idDict: Dictionary of all nodes in interactome, mapped to an integer value for MCF
    """
    
    #G = min_cost_flow.SimpleMinCostFlow()
    G = nx.DiGraph()
    idDict = dict()
    curID = 0
    default_capacity = 1
    
    # Go through edge_file, assign each node an id
    with open(edges_file) as edges_f:
        for line in edges_f:
            tokens = line.strip().split()
            node1 = tokens[0]
            
            # Add nodes to idDict if they aren't there
            if not node1 in idDict:
                idDict[node1] = curID
                curID += 1
            node2 = tokens[1]
            if not node2 in idDict:
                idDict[node2] = curID
                curID += 1
            
            # Convert weight to integer, as google can't handle non-int weights
            w = int(((1 - float(tokens[2])))*100)
            #G.add_arc_with_capacity_and_unit_cost(idDict[node1], idDict[node2], default_capacity, w)
            #G.add_arc_with_capacity_and_unit_cost(idDict[node2], idDict[node1], default_capacity, w)
            
            G.add_edge(idDict[node1], idDict[node2], cost = w, cap = default_capacity)
            
        idDict["maxID"] = curID
        return G, idDict
    
def add_sources_and_targets(G, sources, targets, idDict, flow):
    """
    """
    source_weight = 1/len(sources)
    target_weight = 1/len(targets)
    
    G1 = G.copy()
    
    source_cap = source_weight
    target_cap = target_weight
    
    # subsets capturing the source and target nodes
    gen = []
    tra = []
    
    curID = idDict["maxID"]
    idDict["source"] = curID
    curID += 1
    idDict["target"] = curID

    for source in sources:
        print(source)
        if source in idDict:
            print("found")
            #G.add_arc_with_capacity_and_unit_cost(idDict["source"],idDict[source], source_cap, source_weight)
            G1.add_edge(idDict["source"], idDict[source], cost = source_weight, cap = source_cap)
            gen.append(idDict[source])

    for target in targets:
        print(target)
        if target in idDict:
            #G.add_arc_with_capacity_and_unit_cost(idDict[target],idDict["target"], target_cap, target_weight)
            G1.add_edge(idDict["target"], idDict[target], cost = target_weight, cap = target_cap)
            tra.append(idDict[target])
            
    return G1, gen, tra
    
def prepare_variables(solver, G1, default_capacity):
    flows = dict()
    extras = 0
    for edge in G1.edges():
        if edge not in flows:
            flows[edge] = solver.NumVar(0.0, default_capacity, f"Flows{edge}")
            G1.get_edge_data(edge[0],edge[1])["flow"] = flows[edge]
        else:
            print("repeat")
            print(edge)
            extras += 1
    print(f"We had {extras} repeat edges")
    return flows
    
def prepare_constraints(solver, G1, idDict, flows):
    constraints = []
    for i,  node in enumerate(G1.nodes):
        
        in_edges = list(G1.in_edges(node))
        out_edges = list(G1.out_edges(node))
        
        in_flow = 0
        out_flow = 0
        
        
        constraints.append(solver.Constraint(flow[j] for j in in_edges)
        
        for j in in_edges:
            for k in out_edges:    
                if j[0] == idDict["source"]:
                    print(f"found edge {j} connecting to source")
                elif k[1] == idDict["target"]:
                    print(f"found edge {k} connecting to target")
                else:
                    constraints.append(solver.Constraint())
                    
        print("HI")
        
def prepare_objective(solver, G1, flows):
    objective = solver.Objective()

def responsenet(G, G1, flow, output, idDict, gamma, gen, tra):
    """ The NEW ILP solver for MinCostFlow, using glop.
    """
    
    # temporary value, will make it changable later
    default_capacity = 1

    solver = pywraplp.Solver.CreateSolver("GLOP")
    if not solver:
        return
    
    flows = prepare_variables(solver, G1, default_capacity)
    
    gam = solver.NumVar(gamma, "gam")
    total_flow = solver.NumVar(0, solver.infinity(), "flow")
    
    s,t = idDict["source"], idDict["target"]
    
    solver.Add(0 <= G1.get_edge_data(i,j)[0]["flow"] <= G1.get_edge_data(i,j)[0]["cap"] for i,j in G1.edges())
    solver.Add((sum(G1.get_edge_data(s,i)[0]["flow"] for i in gen)-sum(G1.get_edge_data(j,t) for j in tra)) == 0)
    
    
    for node in idDict:
        for outgoing,incoming in node:
            break
            
    
    
def main():
    
    sources = parse_nodes("sources2.txt")
    targets = parse_nodes("targets2.txt")
    
    G, idDict = construct_digraph("edges2.txt")
    
    G1, gen, tra = add_sources_and_targets(G, sources, targets, idDict, 1)
    
    # Temporary
    solver = pywraplp.CreateSolver("GLOP")
    
    return G, G1, idDict, gen, tra, sources, targets, solver
    
    
    