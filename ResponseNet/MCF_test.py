"""From Bradley, Hax and Maganti, 'Applied Mathematical Programming', figure 8.1."""
import numpy as np

from ortools.graph.python import min_cost_flow


def main():
    """MinCostFlow simple interface example."""
    # Instantiate a SimpleMinCostFlow solver.
    smcf = min_cost_flow.SimpleMinCostFlow()

    #test_smcf = min_cost_flow.SimpleMinCostFlow()
    # Define four parallel arrays: sources, destinations, capacities,
    # and unit costs between each pair. For instance, the arc from node 0
    # to node 1 has a capacity of 15.
    start_nodes = np.array([0, 0, 1, 1, 1, 2, 2, 3, 4])
    end_nodes = np.array([1, 2, 2, 3, 4, 3, 4, 4, 2])
    capacities = np.array([15, 8, 20, 4, 10, 15, 4, 20, 5])
    unit_costs = np.array([4, 4, 2, 2, 6, 1, 3, 2, 3])

    test_snodes = np.array([0,1,1,2,2,3,3,4,4,4,5,5,6,7,7,8,8])
    test_enodes = np.array([3,3,5,4,5,6,5,7,5,8,8,11,9,6,11,9,10])
    test_caps = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    test_ucosts = np.array([9,3,4,9,1,9,7,6,3,2,5,4,7,8,1,3,6])

    print(len(test_snodes), len(test_caps), len(test_enodes),len(test_ucosts))

    # Define an array of supplies at each node.
    supplies = [20, 0, 0, -5, -15]

    test_supplies = [1,1,1,0,0,0,0,0,-1,-1,-1]

    # Add arcs, capacities and costs in bulk using numpy.
    # all_arcs = smcf.add_arcs_with_capacity_and_unit_cost(
    #     start_nodes, end_nodes, capacities, unit_costs
    # )

    all_arcs = smcf.add_arcs_with_capacity_and_unit_cost(
        test_snodes, test_enodes, test_caps, test_ucosts
    )

    # Add supply for each nodes.
    #smcf.set_nodes_supplies(np.arange(0, len(supplies)), supplies)

    smcf.set_nodes_supplies(np.arange(0, len(test_supplies)), test_supplies)

    # Find the min cost flow.
    #status = smcf.solve()
    status = smcf.solve()

    if status != smcf.OPTIMAL:
        print("There was an issue with the min cost flow input.")
        print(f"Status: {status}")
        exit(1)
    print(f"Minimum cost: {smcf.optimal_cost()}")
    print("")
    print(" Arc    Flow /Capacity \t\t Cost")
    solution_flows = smcf.flows(all_arcs)
    # print(f"Debug: {solution_flows} and {unit_costs}")
    costs = solution_flows * test_ucosts
    for arc, flow, cost in zip(all_arcs, solution_flows, costs):
        print(
            f"{smcf.tail(arc):1} -> {smcf.head(arc)}\t{flow:3}  / {smcf.capacity(arc):3}\t\t{cost}"
        )


if __name__ == "__main__":
    main()