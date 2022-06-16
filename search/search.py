# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from sre_constants import SUCCESS
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

class Node:
    def __init__(self, state, pred, action, priority=0):
        self.state = state
        self.pred = pred
        self.action = action
        self.priority = priority
    def __repr__(self):
        return "State: {0}, Action: {1}".format(self.state, self.action)

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST

    return [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start:", problem.getStartState())
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** :YOUR CODE HERE ***"

    from game import Directions
    explored = set()
    fringe = util.Stack()

    if problem.isGoalState(problem.getStartState()):
        return Directions.STOP

    fringe.push([(problem.getStartState(), "Start", 0)]) # path = (state, "state_name", is_explored)

    while not fringe.isEmpty():
        current_path = fringe.pop()
        current_state = current_path[-1][0] 

        if current_state in explored:
            continue

        if problem.isGoalState(current_state):
            break

        successors = problem.getSuccessors(current_state)
        for x in successors:
            if x[0] not in explored:
                next_path = current_path[:]
                next_path.append(x)
                fringe.push(next_path)

        explored.add(current_state)
    
    return  [x[1] for x in current_path[1:]]

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from game import Directions
    explored = set()
    fringe = util.Queue()

    if problem.isGoalState(problem.getStartState()):
        return Directions.STOP

    start_state = problem.getStartState() 
    start_path = [(start_state, "Start", 0)]
    
    fringe.push(start_path)
    explored.add(start_state)

    while not fringe.isEmpty():
        current_path = fringe.pop()
        current_state = current_path[-1][0]

        if problem.isGoalState(current_state): break

        successors = problem.getSuccessors(current_state)

        for x in successors:
            if x[0] in explored: continue

            if x[0] not in explored:
                next_path = current_path[:]
                next_path.append(x)

                fringe.push(next_path)
                explored.add(x[0])

    return [x[1] for x in current_path[1:]]

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from game import Directions
    closed = set()
    
    fringe = util.PriorityQueue()
    fringe.push(Node(problem.getStartState(), None, None), 0)

    while fringe.isEmpty() is not True:
        node = fringe.pop()
        if problem.isGoalState(node.state) is True:
            actions = list()

            while node.action is not None: # Backtracking actions list
                actions.append(node.action)
                node = node.pred
            actions.reverse()
            return actions

        if node.state not in closed:
            closed.add(node.state)
            for s in problem.getSuccessors(node.state):
                fringe.push(Node(s[0], node, s[1], s[2]+node.priority),\
                            s[2]+node.priority)
    return list()

    #util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from searchAgents import manhattanHeuristic
    from game import Directions

    #heuristic = manhattanHeuristic
    start_state = problem.getStartState()
    start_path = [(start_state, "Start", 0)] 

    if problem.isGoalState(start_state): return Directions.STOP

    explored = set()
    explored.add(start_state)

    fringe = util.PriorityQueue()
    fringe.push(start_path, 0)

    while not fringe.isEmpty():
        curr_path = fringe.pop()
        curr_state = curr_path[-1][0]
        curr_actions = [x[1] for x in curr_path[1:]]

        if problem.isGoalState(curr_state): break

        curr_cost = problem.getCostOfActions(curr_actions)
        successors = problem.getSuccessors(curr_state)

        for x in successors:
            if x[0] in explored: continue

            else:
                next_path = curr_path[:]
                next_path.append(x)
                next_cost = curr_cost + x[2] + heuristic(x[0], problem)
                
                fringe.update(next_path, next_cost)
                explored.add(x[0])
    
    return curr_actions
    
    #util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
