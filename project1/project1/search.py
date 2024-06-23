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


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    # initialize stack and set of visited states
    stack = util.Stack()
    visited = []
    # get start state
    start_state = problem.getStartState()
    # check if goal state
    if problem.isGoalState(start_state):
        return []
    # add to stack
    stack.push((start_state,[]))
    # start algorithm
    while not stack.isEmpty():
        # take new action
        current, actions_taken = stack.pop()
        if current not in visited:
            visited.append(current)
            # check if goal state
            if problem.isGoalState(current):
                return actions_taken # return list of actions
            #get children
            successors = problem.getSuccessors(current)
            for state, action, cost in successors:
                newPath = actions_taken + [action]
                stack.push((state, newPath))

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    # initialize queue and visited states set
    queue = util.Queue()
    visited = []
    # get starter state and do starter checks
    start_state = problem.getStartState()
    # check if goal
    if problem.isGoalState(start_state):
        return start_state[1]
    queue.push((start_state,[]))
    # start search
    while queue.isEmpty() is False:
        current, actions_taken = queue.pop()
        if current not in visited:
            visited.append(current)
            #check if goal state
            if problem.isGoalState(current):
                return actions_taken
            # get sucessors
            sucessors = problem.getSuccessors(current)
            for state, action, cost in sucessors:
                queue.push((state, actions_taken + [action]))
    return[]

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    #initialize data structs
    priq = util.PriorityQueue()
    visited = []
    path = []
    
    #check if goal
    start = problem.getStartState()
    if problem.isGoalState(start):
        return []
    
    #start from initial state with empty path and cost of 0
    priq.push((start,[]),0)
    
    while not priq.isEmpty():
        pos, actions = priq.pop() # get position and current path
        visited.append(pos)
        
        #check for goal
        if problem.isGoalState(pos):
            return actions
        
        #get children
        successors = problem.getSuccessors(pos)
        for succ in successors:
            #if not in visited and not already stored in the heap -> we push to heap
            if succ[0] not in visited and (succ[0] not in (state[2][0] for state in priq.heap)):
                updatedPath = actions + [succ[1]]
                pathCost = problem.getCostOfActions(updatedPath)
                priq.push((succ[0],updatedPath),pathCost)
                
            #if not in visited but already stored in heap -> we check if we need to update heap
            elif succ[0] not in visited and (succ[0] in (state[2][0] for state in priq.heap)):
                for state in priq.heap:
                    if state[2][0] == succ[0]:
                        oldPath = problem.getCostOfActions(state[2][1])
                updatedPath = problem.getCostOfActions(actions + [succ[1]])
                # we compare cost of the current path to the new proposed path, if new path is cheaper, update
                if oldPath > updatedPath:
                    updatedPath = actions + [succ[1]]
                    priq.update((succ[0], updatedPath), pathCost)
            
    
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    queue = util.PriorityQueue()
    visited = []
    path = []
    start_state = problem.getStartState()
    #check if goal
    if problem.isGoalState(start_state):
        return []
    queue.push((start_state,[],0),0)
    while not queue.isEmpty():
        current,path,pathcost = queue.pop()
        if current not in visited:
            visited.append(current)
            #check if goal
            if problem.isGoalState(current):
                return path
            #expand
            successors = problem.getSuccessors(current)
            for state, action, step_cost in successors:
                updatedPath = path + [action]
                newCost = pathcost + step_cost
                heurCost = newCost + heuristic(state,problem)
                queue.push((state,updatedPath,newCost), heurCost)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
