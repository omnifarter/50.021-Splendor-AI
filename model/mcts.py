# References:
# https://github.com/asdfjkl/neural_network_chess
import copy
import itertools
import logging
import operator
import random
import time
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
    result: typing.Optional[int]

    def __init__(self, board, parent_edge, result=None):
        self.board = board
        self.parentEdge = parent_edge
        self.childEdgeNode = []
        self.result = result

    def expand(self, eval_fn: Callable[[Any], Tuple[List[Number], Number]]):
        """
        @param eval_fn: a function that takes in a board state and returns a tuple of (policy, evaluation)
        @return: the valuation of the position
        """
        moves = self.board.possible_actions()
        for m in moves:
            child_board = copy.deepcopy(self.board)
            reward, done = child_board.playerAction(m)
            edge = Edge(m, self)
            child = Node(child_board, edge)
            if reward == 3: # p1 win
                winner = 1
            elif reward == -1:
                winner = -1
            self.childEdgeNode.append((edge, child))

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
        if node.result is not None:
            v = node.result
            self.backprop(v, node.parentEdge)
            return
        v = node.expand(eval_fn=self.eval_fn)
        self.backprop(v, node.parentEdge)

    def backprop(self, v, edge):
        edge.N += 1
        edge.W += edge.W + v
        edge.Q = edge.W / edge.N
        if edge.parentNode is not None:
            if edge.parentNode.parentEdge is not None:
                self.backprop(v, edge.parentNode.parentEdge)

    def search(self, rootNode, select_timeout=1):
        # timeout in seconds
        self.rootNode = rootNode
        _ = self.rootNode.expand(self.eval_fn)
        selected_node = None
        t = time.time()
        i = 0
        while time.time() < t + select_timeout:
            i += 1
            selected_node = self.select(rootNode)
            self.expand_and_eval(selected_node)
        logging.info(f"{i} iterations")
        N_sum = 0
        moveProbs = []
        for edge, _ in rootNode.childEdgeNode:
            N_sum += edge.N
        for (edge, node) in rootNode.childEdgeNode:
            prob = (edge.N ** (1 / self.tau)) / ((N_sum) ** (1 / self.tau))
            moveProbs.append((edge.move, prob, edge.N, edge.Q))
        return moveProbs


def random_eval(iter=1, max_moves=100):
    def eval_fn(board: rl_rules.Board):
        board = copy.deepcopy(board)
        policy = [1]*28
        out = 0
        for i in range(iter):
            for j in range(max_moves):
                logging.info(board)
                reward, done = board.playerAction(random.choice(board.possible_actions()))
                if done:
                    out += np.sign(reward)
                    break
        return policy, out/iter
    return eval_fn


if __name__ == "__main__":
    board = rl_rules.Board()
    for i in itertools.count():
        mcts = MCTS(random_eval())
        edge = Edge(None, None)
        node = Node(board, edge)
        move_probs = mcts.search(node)
        move = max(move_probs, key=lambda t: t[1])
        print(board)
        print(board.human_action_description(move[0]))
        reward, done = board.playerAction(move[0])
        if done:
            print(f"Player {0 if reward > 0 else 1} wins after {i//2 + 1} moves!")
            break
