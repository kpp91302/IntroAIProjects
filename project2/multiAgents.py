# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        '''
        The eval function needs to the following basic things:
        1. keep the agent away from the ghost agent to get full points
        2. increase the score each turn in order to get full points
        '''
        # get the closest ghost in the game
        min_safe_dist = 100
        for state in newGhostStates:
            ghost_dist = util.manhattanDistance(newPos, state.getPosition())
            if ghost_dist <= min_safe_dist:
                min_safe_dist = ghost_dist
        
        '''
        This section is used to find the distance to the closest ghost
        in the sucessor state and, the distance to the closest pellet in both current
        and successor state.
        
        We will use these to help determine which actions to reward or penalize
        '''
        current_position = currentGameState.getPacmanPosition()
        
        # find the distance to closet pellet
        closest_pellet = 1000
        for pellet in currentGameState.getFood().asList():
            distance = util.manhattanDistance(current_position, pellet)
            if distance <= closest_pellet:
                closest_pellet = distance
        # get the closest pellet in the sucessor state
        successor_pellets = successorGameState.getFood().asList()
        closest_successor_pellet = 1000
        if(len(successor_pellets) == 0): #if the successor state has no pellets that means this action is the win state
            closest_successor_pellet = 0
        else:
            for pellet in successor_pellets:
                distance = util.manhattanDistance(successorGameState.getPacmanPosition(), pellet)
                if distance <= closest_successor_pellet:
                    closest_successor_pellet = distance
        
        # get the difference in closest pellet for current and successor state
        closest_pellet_difference = closest_pellet - closest_successor_pellet
        
        '''
        This section is where we reward/penalize certain actions
        some considerations are the following:
        - We want to keep moving since the score decreases every action we take
          if we aren't eating pellets, power pellets, or scared ghosts
        - power pellets do no contribute towards reaching win state so we dont want to reward
          actions that will lead to pacman wasting time trying to eat a scared ghost
        - we want to avoid random movements if the closest pellet is a tie
        '''
        if min_safe_dist <= 1 or action == Directions.STOP:
            return 0 #stay away from the ghosts (not too far) and dont stop moving
        if (successorGameState.getScore() - currentGameState.getScore()) > 0:
            return 10 #prioritize higher scores
        elif closest_pellet_difference > 0:
            return 5 #if the score doesnt increase then go to closet pellet
        elif action == currentGameState.getPacmanState().getDirection():
            return 2 #if the distance to closet pellet is a tie then go same direction
        else:
            return 1 #all other cases
        
        

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        #helper functions for min and max actions of MiniMax algorithm
        def min_action(state, agentIndex, depth):
            # need to get legal actions for each action
            legal = state.getLegalActions(agentIndex)
            #if there are no legal actions then game is over
            if not legal:
                return self.evaluationFunction(state)

            '''
            from this point we will recursively call min_action for each ghost
            We will do this until the agent is pacman
            When it is pacman we will call max_action
            '''
            if agentIndex == state.getNumAgents() - 1:
                max_action_values = [max_action(state.generateSuccessor(agentIndex, action), depth) for action in legal]
                return min(max_action_values)
            else: 
                min_value_actions = [min_action(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth) for action in legal]
                return min(min_value_actions)
            
        #only pacman will try to maximize so no need for agentIndex
        def max_action(state, depth):
            legal = state.getLegalActions(0)
            if not legal or depth == self.depth:
                return self.evaluationFunction(state)
            min_action_values = [min_action(state.generateSuccessor(0, action), 1, depth + 1) for action in legal]
            return max(min_action_values)
        
        #Main Algorithm
        
        # Initialize variables to keep track of the best action and its corresponding value
        pacman_action = None
        best = float('-inf')  # Initialize to negative infinity

        # Iterate through legal actions for Pacman
        for action in gameState.getLegalActions(0):
            # Calculate the value using the minValue function
            value = min_action(gameState.generateSuccessor(0, action), 1, 1)
            # Update the best action and best value if the current value is greater
            if value > best:
                best = value
                pacman_action = action

        # Return the best action
        return pacman_action
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # max level is called by pacman agent, agentIndex is always 0 for this function
        def max_level(gameState, depth, alpha, beta):
            new_depth = depth + 1
            if gameState.isWin() or gameState.isLose() or new_depth == self.depth:
                return self.evaluationFunction(gameState)
            max_val = -999999
            legal = gameState.getLegalActions(0)
            current_alpha = alpha
            for action in legal:
                max_val = max(max_val, min_level(gameState.generateSuccessor(0, action), new_depth, 1, current_alpha, beta))
                if max_val > beta:
                    return max_val
                current_alpha = max(current_alpha, max_val)
            return max_val
        # ghosts call min_level so we need agentIndex
        def min_level(gameState, depth, agentIndex, alpha, beta):
            min_val = 999999
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            legal = gameState.getLegalActions(agentIndex)
            current_beta = beta
            for action in legal:
                if agentIndex == (gameState.getNumAgents() - 1):
                    min_val = min(min_val, max_level(gameState.generateSuccessor(agentIndex, action), depth , alpha, current_beta))
                    if min_val < alpha:
                        return min_val
                    current_beta = min(current_beta, min_val)
                else:
                    min_val = min(min_val, min_level(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1, alpha, current_beta))
                    if min_val < alpha:
                        return min_val
                    current_beta = min(current_beta, min_val)
            return min_val
            
        '''
        main algorithm
        '''
        legal = gameState.getLegalActions(0)
        score = -999999
        return_action = ''
        # set alpha and beta to positive and negative infinity
        alpha = float('-inf')
        beta = float('inf')
        for action in legal:
            nextState = gameState.generateSuccessor(0, action)
            for_score = min_level(nextState, 0, 1, alpha, beta)
            if for_score > score:
                return_action = action
                score = for_score
            if for_score > beta:
                return return_action
            alpha = max(alpha, for_score)
        return return_action
        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        #Used only for pacman agent hence agentindex is always 0.
        def maxLevel(gameState,depth):
            currDepth = depth + 1
            if gameState.isWin() or gameState.isLose() or currDepth==self.depth:   #Terminal Test 
                return self.evaluationFunction(gameState)
            maxvalue = -999999
            actions = gameState.getLegalActions(0)
            totalmaxvalue = 0
            numberofactions = len(actions)
            for action in actions:
                successor= gameState.generateSuccessor(0,action)
                maxvalue = max (maxvalue,expectLevel(successor,currDepth,1))
            return maxvalue
        
        #For all ghosts.
        def expectLevel(gameState,depth, agentIndex):
            if gameState.isWin() or gameState.isLose():   #Terminal Test 
                return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(agentIndex)
            totalexpectedvalue = 0
            numberofactions = len(actions)
            for action in actions:
                successor= gameState.generateSuccessor(agentIndex,action)
                if agentIndex == (gameState.getNumAgents() - 1):
                    expectedvalue = maxLevel(successor,depth)
                else:
                    expectedvalue = expectLevel(successor,depth,agentIndex+1)
                totalexpectedvalue = totalexpectedvalue + expectedvalue
            if numberofactions == 0:
                return  0
            return float(totalexpectedvalue)/float(numberofactions)
        
        
        #Root level action.
        actions = gameState.getLegalActions(0)
        currentScore = -999999
        returnAction = ''
        for action in actions:
            nextState = gameState.generateSuccessor(0,action)
            # Next level is a expect level. Hence calling expectLevel for successors of the root.
            score = expectLevel(nextState,0,1)
            # Choosing the action which is Maximum of the successors.
            if score > currentScore:
                returnAction = action
                currentScore = score
        return returnAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    
    ''' Retrieve relevant game state data the pacman agent will need'''
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    """ get the distance from the pacman agent in the sucessor state to each food pellet using manhattan distance"""
    pellets = newFood.asList()
    pellet_distances = [0]
    for pos in pellets:
        pellet_distances.append(manhattanDistance(newPos,pos))
    
    """ get the distance from the pacman agent in the sucessor state to each ghost using manhattan distance """
    ghostPos = []
    ghostDistance = [0]
    for ghost in newGhostStates:
        ghostPos.append(ghost.getPosition())
    for pos in ghostPos:
        ghostDistance.append(manhattanDistance(newPos,pos))

    """ get number of power capsules available in the current game state """
    PowerCaps = len(currentGameState.getCapsules())
    
    """ Algorithm for Eval Function """
    #set score variable to return a evaulation to the pacman agent
    # get amount of food, how much longer the ghosts are eatable (if applicable), and the total distance away from the ghosts
    score = 0
    remaining_food = len(newFood.asList(False))           
    scared_time_remaining = sum(newScaredTimes)
    sumGhostDistance = sum (ghostDistance)
    # We want the pacman agent to priotitize chasing an area with multiple pellets rather than the single closest pellet
    reciprocalDistance = 0
    
    if sum(pellet_distances) > 0:
        reciprocalDistance = 1.0 / sum(pellet_distances)
    
    """we increment the score based on the reciprocal rather than actual distance to 
    incentivize going for clusters of pellets"""
    score += currentGameState.getScore()  + reciprocalDistance + remaining_food

    # if the ghosts are scared we want to prioritize eating close ghosts
    if scared_time_remaining > 0:    
        score +=   scared_time_remaining + (-1 * PowerCaps) + (-1 * sumGhostDistance)
        return score # we no longer need to consider other factors
    
    #if the ghosts are not scared we want to stay away from the ghosts and not prioritize the power capsules
    #since the capsules do not effect score significantly
    score +=  sumGhostDistance + PowerCaps
    return score

# Abbreviation
better = betterEvaluationFunction
