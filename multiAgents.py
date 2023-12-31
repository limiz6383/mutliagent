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
        min_ghost_dist = float("inf")
        for state in newGhostStates:
            if state.scaredTimer == 0:
                g_x, g_y = state.getPosition()
                new_dist = manhattanDistance(newPos, (g_x, g_y))
                min_ghost_dist = min(new_dist, min_ghost_dist)
        
        foodlist = newFood.asList()
        min_food_dist = float("inf")
        if not foodlist:
            min_food_dist = 0
        else:
            for food in foodlist:
                new_dist = manhattanDistance(newPos, food)
                min_food_dist = min(min_food_dist, new_dist)
        
        weightedghostdist = 1/ (min_ghost_dist + 0.3) 
        weightedfooddist = 1/ (min_food_dist + 0.5) 
        currscore = successorGameState.getScore() ** 2
        score = weightedghostdist + weightedfooddist + currscore
        if min_ghost_dist > 0:
            score -= 10/min_ghost_dist
        if len(foodlist):
            score += 10/min_food_dist
        return score

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
        actions = gameState.getLegalActions(0)
        idx = 0 
        maxidx = 0 
        maxscore = float("-inf")
        for action in actions:
            newstate = gameState.generateSuccessor(0, action)
            newscore = self.value(newstate, 1, 0)
            if newscore > maxscore:
                maxscore= newscore
                maxidx = idx
            idx += 1
        return actions[maxidx]
    
    def value(self, state, agent, depth):
        if self.depth == depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        if agent == 0: 
            return self.maxValue(state, agent, depth)
        if agent > 0:
            return self.minValue(state, agent, depth)

    def maxValue(self, state: GameState, agent, depth):
        v = float("-inf")
        actions = state.getLegalActions(agent)
        for action in actions:
            s = state.generateSuccessor(agent, action)
            v = max(self.value(s, agent + 1, depth), v)
        return v
    
    def minValue(self, state: GameState, agent, depth):
        v = float("inf")
        actions = state.getLegalActions(agent)
        for action in actions:
            s = state.generateSuccessor(agent, action)
            numGhosts = state.getNumAgents() - 1
            if numGhosts == agent:
                v = min(self.value(s, 0, depth + 1), v)
            else:
                v = min(v, self.value(s, agent + 1, depth))
        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        alpha = float("-inf")
        beta = float("inf")
        idx = 0 
        maxidx = 0 
        maxscore = float("-inf")
        for action in actions:
            newstate = gameState.generateSuccessor(0, action)
            newscore = self.value(newstate, alpha, beta, 1, 0)
            if newscore > maxscore:
                maxscore= newscore
                maxidx = idx
            alpha = max(alpha, newscore)
            idx += 1
            
        return actions[maxidx]
    
    def value(self, state, alpha, beta, agent, depth):
        if self.depth == depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        if agent == 0: 
            return self.maxValue(state, alpha, beta, agent, depth)
        if agent > 0:
            return self.minValue(state, alpha, beta, agent, depth)

    def maxValue(self, state, alpha, beta, agent, depth):
        v = float("-inf")
        actions = state.getLegalActions(agent)
        for action in actions:
            s = state.generateSuccessor(agent, action)
            v = max(v, self.value(s, alpha, beta, agent + 1, depth))
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v
    
    def minValue(self, state, alpha, beta, agent, depth):
        v = float("inf")
        actions = state.getLegalActions(agent)
        for action in actions:
            s = state.generateSuccessor(agent, action)
            numGhosts = state.getNumAgents() - 1

            if numGhosts == agent:
                v = min(self.value(s, alpha, beta, 0, depth + 1), v)
            else:
                v = min(v, self.value(s, alpha, beta, agent + 1, depth))
            if v < alpha:
                return v
            beta = min(beta, v)
        return v


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
        actions = gameState.getLegalActions(0)
        idx = 0 
        maxidx = 0 
        maxscore = float("-inf")
        for action in actions:
            newstate = gameState.generateSuccessor(0, action)
            newscore = self.value(newstate, 1, 0)
            if newscore > maxscore:
                maxscore= newscore
                maxidx = idx
            idx += 1
        return actions[maxidx]
    
    def value(self, state, agent, depth):
        if self.depth == depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        if agent == 0:
            return self.maxValue(state, agent, depth)
        if agent > 0:
            return self.expValue(state, agent, depth)

    def maxValue(self, state: GameState, agent, depth):
        v = float("-inf")
        actions = state.getLegalActions(agent)
        for action in actions:
            s = state.generateSuccessor(agent, action)
            v = max(self.value(s, agent + 1, depth), v)
        return v
    
    def expValue(self, state: GameState, agent, depth):
        v = 0
        actions = state.getLegalActions(agent)
        for action in actions:
            s = state.generateSuccessor(agent, action)
            numGhosts = state.getNumAgents() - 1
            if numGhosts == agent:
                v += self.value(s, 0, depth + 1)
            else:
                v += self.value(s, agent + 1, depth)
        return v


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    "*** YOUR CODE HERE ***"
    min_ghost_dist = float("inf")
    max_ghost_dist = float("-inf")
    dist_from_ghosts = []
    for state in ghostStates:
        g_x, g_y = state.getPosition()
        new_dist = manhattanDistance(pos, (g_x, g_y))
        dist_from_ghosts.append(new_dist)    
    scared = sum([10 / (d + 0.1) for d in dist_from_ghosts])
    notScared = sum([20 / ((d + 0.1) ** 5) for d in dist_from_ghosts])
    ghost_dist = min(dist_from_ghosts) 
    
    foodlist = foods.asList()
    food_list = []
    min_food_dist = float("inf")
    foodAdd = 0
    if not foodlist:
        food_list = 0
    else:
        for food in foodlist:
            new_dist = manhattanDistance(pos, food)
            food_list.append(new_dist)
            min_food_dist = min(min_food_dist, new_dist)
        foodAdd = sum(1/(f ** 2 + 0.1) for f in food_list) ** 2
        
    weightedghostdist = 1 / (ghost_dist ** 2 + 0.5) 
    weightedfooddist = 1 / (min_food_dist ** 4 + 0.5) 
    pacman = currentGameState.getPacmanState()
    currscore = currentGameState.getScore() ** 2
    score = weightedghostdist + weightedfooddist + currscore - notScared + scared + foodAdd
    if min_ghost_dist > 0:
        score -= 5 / (ghost_dist + 0.1)
    if len(foodlist):
        score += 5 / (min_food_dist + 0.1)
    if pacman.getDirection() == Directions.STOP:
        score -= 1 / (score ** 10)
    return score

# Abbreviation
better = betterEvaluationFunction
