import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
    
        TODO: Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """

        graph = self.adj_mat #Easy variable for adjacency matrix
        N = len(graph)  #Getting N, or number of nodes

        #Initializing mst with zeros, empty set visited nodes, and an empty queue of neighbors
        mst = np.zeros((N,N))
        visited_nodes = set()
        neighbors = []
        heapq.heapify(neighbors)

        #Picking A random Node 
        start_node = np.random.randint(0, N)
        visited_nodes.add(start_node)


        for i, value in enumerate(graph[start_node]):
            if value > 0 and i not in visited_nodes:  # do not want to push neighbers with edge length 0, those are self or not connected
                heapq.heappush(neighbors, (value, start_node, i))

        #Begin loop, break from the loop once the number of visited nodes is no longer less than the number of nodes
        while len(visited_nodes) < N:
            #Extract edge length from value, index of edge length from i, and index of parent node from parent
            value, parent, i = heapq.heappop(neighbors)
            if i in visited_nodes:
                continue
            
            #Add parent to visited nodes, this will prevent loops within the mst
            visited_nodes.add(i)

            #Add edge length to correct index in mst
            mst[parent,i] = value
            mst[i, parent] = value

            #Now iterating over index and value of new node
            for j, value in enumerate(graph[i]):
                if value > 0 and j not in visited_nodes: #Filters out no connections and self connections
                    heapq.heappush(neighbors, (value, i, j)) #Stores new neighbor edge length, the new parent node, and the index of the edge lengths
        self.mst = mst