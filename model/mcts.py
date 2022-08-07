# References:
# https://github.com/asdfjkl/neural_network_chess
import random
import typing
from numbers import Number
from typing import Any, Callable, List, Tuple

import numpy as np

from model import rl_rules


class Edge:
    """
    A class for representing edges in a Monte Carlo search tree
\
    @ivar parentNode: the parent node
    @ivar move: the action taken from the parent node
    @ivar N: the number of times the edge has been visited
    @ivar Q: the average game result after action A
    @ivar P: the prior probabilities from our network/heuristics
    @ivar W: estimated reward of the node
    """
    parentNode: "Node"

    def __init__(self, move, parentNode):
        self.parentNode = parentNode
        self.move = move
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = 0


class Node:
    childEdgeNode: List[Tuple[Edge, "Node"]]
    parentEdge: Edge
    board: rl_rules.Board


    def __init__(self, board, parent_edge):
        self.board = board
        self.parentEdge = parent_edge
        self.childEdgeNode = []

    def expand(self, eval_fn: Callable[[Any], Tuple[List[Number], Number]]):
        """
        @param eval_fn: a function that takes in a board state and returns a tuple of (policy, evaluation)
        @return: the valuation of the position
        """
        # TODO get all moves
        moves = range(26)
        for m in moves:
            # TODO create a new child node, append to childEdgeNode list
            pass
        # run eval on current position to determine policies for moves
        policy, evaluation = eval_fn(self.board)
        prob_sum = 0.
        for edge, _ in self.childEdgeNode:
            edge.P = policy[edge.move]
            prob_sum += edge.P
        for edge,_ in self.childEdgeNode:
            edge.P /= prob_sum
        return evaluation
        

    def is_leaf(self):
        return self.childEdgeNode == []

class MCTS:
    def __init__(self, eval_fn):
        self.eval_fn = eval_fn
        self.rootNode = None
        self.tau = 1.0
        self.c_puct = 1.0

    def uct_value(self, edge: Edge, parent_n: Number):
        return self.c_puct * edge.P * np.sqrt(parent_n) / (1 + edge.N)
    
    def select(self, node: "Node"):
        if node.is_leaf():
            return node
        
        max_uct_children = []
        max_uct_value = float("-inf")
        
        for edge, child in node.childEdgeNode:
            uct_val = self.uct_value(edge, edge.parentNode.parentEdge.N)
            val = edge.Q
            # if black to move: reverse directions
            if edge.parentNode.board.current_player == edge.parentNode.board.player2:
                val = -edge.Q
            uct_val_child = val + uct_val
            if uct_val_child > max_uct_value:
                max_uct_value = uct_val_child
                max_uct_children = [child]
            elif uct_val_child == max_uct_value:
                max_uct_children.append(child)
        if not max_uct_children:
            raise ValueError("could not find child with best value")
        else:
            return self.select(random.choice(max_uct_children))

    def expand_and_eval(self, node: Node):
        pass

def random_eval(board: rl_rules.Board):
    policy = [1]*24