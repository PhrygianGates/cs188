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

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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

    def evaluationFunction(self, currentGameState, action):
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
 
        foodPos = currentGameState.getFood().asList()
        ghostPos = successorGameState.getGhostPositions()
        if newPos in foodPos:
            return 1
        for ghostState in newGhostStates:
            if ghostState.getPosition() == newPos and ghostState.scaredTimer <= 0:
                return -1
        foodDis = min(util.manhattanDistance(newPos, food) for food in foodPos)
        ghostDis = min(util.manhattanDistance(newPos, ghost) for ghost in ghostPos)
        return (1 / foodDis - 1 / ghostDis)

        """
         # return Value [-1,1]

        newFood = newFood.asList()
        ghostPos = [(G.getPosition()[0], G.getPosition()[1]) for G in newGhostStates]
        scared = min(newScaredTimes) > 0

        # if not new ScaredTimes new state is ghost: return lowest value

        if not scared and (newPos in ghostPos):
            return -1.0

        if newPos in currentGameState.getFood().asList():
            return 1

        closestFoodDist = sorted(newFood, key=lambda fDist: util.manhattanDistance(fDist, newPos))
        closestGhostDist = sorted(ghostPos, key=lambda gDist: util.manhattanDistance(gDist, newPos))

        fd = lambda fDis: util.manhattanDistance(fDis, newPos)
        gd = lambda gDis: util.manhattanDistance(gDis, newPos)

        return 1.0 / fd(closestFoodDist[0]) - 1.0 / gd(closestGhostDist[0])

        # return successorGameState.getScore()
        """
def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        def terminate(state, depth):
            return state.isWin() or state.isLose() or depth > self.depth
        
        def max_value(state, depth):
            if terminate(state, depth):
                return self.evaluationFunction(state)
            v = -float('inf')
            for action in state.getLegalActions(0):
                v = max(v, min_value(state.generateSuccessor(0, action), 1, depth))
            return v

        def min_value(state, index, depth):
            if terminate(state, depth):
                return self.evaluationFunction(state)
            v = float('inf')
            for action in state.getLegalActions(index):
                #print("index:",index,"num:",state.getNumAgents())
                if index == state.getNumAgents() - 1:
                    v = min(v, max_value(state.generateSuccessor(index, action), depth + 1))
                else:
                    v = min(v, min_value(state.generateSuccessor(index, action), index + 1, depth))
            return v

        results = [(action, min_value(gameState.generateSuccessor(0, action), 1, 1)) for action in gameState.getLegalActions(0)]
        return max(results, key=lambda x: x[1])[0]

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def terminate(state, depth):
            return state.isWin() or state.isLose() or depth > self.depth
        
        def max_value(state, depth, alpha, beta):
            if terminate(state, depth):
                return self.evaluationFunction(state)
            v = -float('inf')
            for action in state.getLegalActions(0):
                v = max(v, min_value(state.generateSuccessor(0, action), 1, depth, alpha, beta))
                if v > beta:
                    return v
                alpha = max(v, alpha)
            return v

        def min_value(state, index, depth, alpha, beta):
            if terminate(state, depth):
                return self.evaluationFunction(state)
            v = float('inf')
            for action in state.getLegalActions(index):
                #print("index:",index,"num:",state.getNumAgents())
                if index == state.getNumAgents() - 1:
                    v = min(v, max_value(state.generateSuccessor(index, action), depth + 1, alpha, beta))
                else:
                    v = min(v, min_value(state.generateSuccessor(index, action), index + 1, depth, alpha, beta))
                if v < alpha:
                    return v
                beta = min(v, beta)
            return v

        def initial(state):
            v = -float('inf')
            alpha = -float('inf')
            beta = float('inf')
            bestAction = None
            for action in state.getLegalActions(0):
                value = min_value(state.generateSuccessor(0, action), 1, 1, alpha, beta)
                if value > v:
                    bestAction, v = action, value
                alpha = max(v, alpha)
            return bestAction
        
        return initial(gameState)
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def terminate(state, depth):
            return state.isWin() or state.isLose() or depth > self.depth
        
        def max_value(state, depth):
            if terminate(state, depth):
                return self.evaluationFunction(state)
            v = -float('inf')
            for action in state.getLegalActions(0):
                v = max(v, exp_value(state.generateSuccessor(0, action), 1, depth))
            return v

        def exp_value(state, index, depth):
            if terminate(state, depth):
                return self.evaluationFunction(state)
            v = 0
            actions = state.getLegalActions(index)
            for action in actions:
                #print("index:",index,"num:",state.getNumAgents())
                if index == state.getNumAgents() - 1:
                    v += max_value(state.generateSuccessor(index, action), depth + 1)
                else:
                    v += exp_value(state.generateSuccessor(index, action), index + 1, depth)
            v /= len(actions)
            return v

        results = [(action, exp_value(gameState.generateSuccessor(0, action), 1, 1)) for action in gameState.getLegalActions(0)]
        return max(results, key=lambda x: x[1])[0]


        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    # Consts
    INF = 100000000.0  # Infinite value
    WEIGHT_FOOD = 10.0  # Food base value
    WEIGHT_GHOST = -10.0  # Ghost base value
    WEIGHT_SCARED_GHOST = 100.0  # Scared ghost base value

    # Base on gameState.getScore()
    score = currentGameState.getScore()

    # Evaluate the distance to the closest food
    distancesToFoodList = [util.manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
    if len(distancesToFoodList) > 0:
        score += WEIGHT_FOOD / min(distancesToFoodList)
    else:
        score += WEIGHT_FOOD

    # Evaluate the distance to ghosts
    for ghost in newGhostStates:
        distance = manhattanDistance(newPos, ghost.getPosition())
        if distance > 0:
            if ghost.scaredTimer > 0:  # If scared, add points
                score += WEIGHT_SCARED_GHOST / distance
            else:  # If not, decrease points
                score += WEIGHT_GHOST / distance
        else:
            return -INF  # Pacman is dead at this point

    return score

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
