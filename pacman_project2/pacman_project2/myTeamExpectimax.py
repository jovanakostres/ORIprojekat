# myTeam.py
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


from pacman_project2.pacman_project2.captureAgents import CaptureAgent
import random, time
from pacman_project2.pacman_project2 import util
from pacman_project2.pacman_project2.game import Directions, Actions
from pacman_project2.pacman_project2 import game
from pacman_project2.pacman_project2.util import nearestPoint
from datetime import datetime


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='TopAgent', second='BottomAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}

    def evaluationFunction(self, currentGameState, action):
        """
        Implementirajte bolju evaluacionu funkciju ovde.

        Evaluaciona funkcija preuzima trenutno stanje i njegove predlozene sledbenike kao tipove
        GameStates (pacman.py) i vraca broj (veci brojevi su bolji).

        Kod ispod izvlaci korisne informacije iz stanja, kao npr. preostalu hranu (newFood)
        i Pacman pozicije nakon pomeranja (newPos).
        newScaredTimes sadrzi broj pokreta za koji ce svaki duh da ostane na svojoj poziciji (scared),
        jer je Pacman pojeo power pellet.

        Istampajte ove varijable kako biste videli sta dobijate i kombinujte ih kako biste napravili
        dobru evaluacionu funkciju.
        """
        # Korisne informacije koje mozete izvuci iz Useful GameState (pacman.py)
        successorGameState = currentGameState.generateSuccessor(self.index, action)
        newPos = successorGameState.getGhostPositions()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        """Calculating distance to the farthest food pellet"""
        newFoodList = newFood.asList()
        min_food_distance = -1
        for food in newFoodList:
            distance = util.manhattanDistance(newPos, food)
            min_food_distance = min(min_food_distance, distance)

        """Calculating the distances from pacman to the ghosts. Also, checking for the proximity of the ghosts (at distance of 1) around pacman."""
        distances_to_ghosts = 1
        proximity_to_ghosts = 0
        for ghost_state in successorGameState.getGhostPositions():
            distance = util.manhattanDistance(newPos, ghost_state)
            distances_to_ghosts += distance
            if distance <= 1:
                proximity_to_ghosts += 1

        """Combination of the above calculated metrics."""
        return successorGameState.getScore() + (1 / float(min_food_distance)) - (
                1 / float(distances_to_ghosts)) - proximity_to_ghosts

        # return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
      Ova default evaluaciona funkcija samo vraca skor stanja.
      Skor je isti onaj koji je ispisan na Pacman GUI-ju.

      Ova funkcija se koristi za agente sa protivnikom.
    """
    return currentGameState.getScore()


class ExpectimaxAgent(ReflexCaptureAgent):

    # def __init__(self, index, timeForComputing = .1):
    #    self.index = 0  # Pacman se uvek nalazi na indeksu 0
    #    self.evaluationFunction = util.lookup('scoreEvaluationFunction', globals())
    #    self.depth = 2

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        self.favoredY = 0.0
        CaptureAgent.registerInitialState(self, gameState)

    def getWeights(self, gameState, action):
        return {'invaderDistance': -10, 'distanceToFood': -1, 'attack-ghosts': -100, 'escape': -30, 'stop': -5}

    def getFeatures(self, gameState, action):
        features = util.Counter()
        # get the sucessor for the current game state and action
        successor = self.getSuccessor(gameState, action)

        # getting the state and the position
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # getting the coordinates
        xPos, yPos = myState.getPosition()
        # getting relative distance
        relX, relY = Actions.directionToVector(action)
        # next move ahead
        nextX, nextY = int(xPos + relX), int(yPos + relY)

        '''
        dividing the maze to two portions based on agent id
        it can be one or two
        '''
        if self.index > 2:
            section = 1
        else:
            section = self.index + 1

        # get opponent food in the list
        food = self.getFood(gameState)
        appendFood = []
        foodList = food.asList()
        # get the walls around
        walls = gameState.getWalls()

        # is your agent PacMan?
        isPacman = myState.isPacman
        # gets all your opponents
        opponents = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        # gets the opponents pacman in your zone
        invaders = [a for a in opponents if a.isPacman and a.getPosition() != None]
        # gets the ghosts your opponents zone
        chasers = [a for a in opponents if not (a.isPacman) and a.getPosition() != None]

        # checking the food and splitting the maze into two for the team
        if len(foodList) > 0:
            for foodX, foodY in foodList:
                if (foodY > section * walls.height / 2 and foodY < (section + 1) * walls.height / 2):
                    appendFood.append((foodX, foodY))
            if len(appendFood) == 0:
                appendFood = foodList
            if min([self.getMazeDistance(myPos, food) for food in appendFood]) is not None:
                features['distanceToFood'] = float(min([self.getMazeDistance(myPos, food) for food in appendFood])) / (
                        walls.width * walls.height)

        '''
        checking the invaders length and if lenght greater than zero finds the 
        minimum distance to attack the invader
        '''
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        '''
        checking if there is are chasers around
        '''
        if len(chasers) > 0 and successor.getAgentState(self.index).isPacman:
            for ghosts in chasers:
                if (nextX, nextY) == ghosts.getPosition():
                    '''
                    checks if the next position leads to ghosts position and 
                    attacks the ghost if it is in edible mode
                    '''
                    if ghosts.scaredTimer > 0:
                        features['attack-ghosts'] += -10
                    # moves away from chasers
                    else:
                        features['escape'] += 1
                        features['distanceToFood'] = 0

                elif (nextX, nextY) in Actions.getLegalNeighbors(ghosts.getPosition(), walls):
                    if ghosts.scaredTimer > 0:
                        features['ignore'] += 1
                    elif isPacman:
                        features['escape'] += 1

        # if action is stop then set the feature stop
        if action == Directions.STOP: features['stop'] = 1

        return features

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        # values = [self.expectimax(1, 0, gameState.generateSuccessor(0, a)) for a in actions]

        # values = [self.evaluate(gameState, a) for a in actions]
        print(self.index)

        values = [self.expectiMax(gameState.generateSuccessor(self.index, a), self.index + 1, 0)[0] for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        # maximum = float("-inf")
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        # bestActions = []

        foodLeft = len(self.getFood(gameState).asList())

        # if foodLeft <= 2:
        #    bestDist = 9999
        #    for action in actions:
        #        utility = self.expectimax(1, 0, gameState.generateSuccessor(0, action))
        #
        #              if utility > maximum:
        #                  maximum = utility
        #                  bestAction = action
        #          return bestAction
        bestAction = random.choice(bestActions)
        return bestAction
        '''
        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction
  
  
        maximum = float("-inf")
        #bestAction = Directions.WEST
        #print(self.index)
        for action in actions:
            value = self.evaluate(gameState, action)
            utility = self.expectimax(1, 0, gameState.generateSuccessor(self.index, action))
            if utility+value > maximum:
                maximum = utility
                bestAction = action
        return  bestAction
  
  
  
  
  
        maximum = float("-inf")
        bestAction = actions[0]
        #if foodLeft <= 2:
        for agentState in actions:
            utility = self.expectimax(self.index+1, 0, gameState.generateSuccessor(self.index, agentState))
            #print(utility)
            #print(agentState)
            if utility > maximum:
                maximum = utility
                bestAction = agentState
        return  bestAction
        '''

    def betterEvaluationFunction(self, currentGameState):

        newPos = currentGameState.getAgentState(self.index).getPosition()
        # myPos = successor.getAgentState(self.index).getPosition()
        newFood = self.getFood(currentGameState).asList()
        newGhostStates = self.getOpponents(currentGameState)
        newCapsules = currentGameState.getCapsules()
        # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        closestGhost = min(
            [self.getDistance(newPos, currentGameState.getAgentState(ghost).getPosition()) for ghost in newGhostStates])
        if newCapsules:
            closestCapsule = min([self.getDistance(newPos, caps) for caps in newCapsules])
        else:
            closestCapsule = 0

        if closestCapsule:
            closest_capsule = -3 / closestCapsule
        else:
            closest_capsule = 100

        if closestGhost:
            ghost_distance = -2 / closestGhost
        else:
            ghost_distance = -500

        foodList = self.getFood(currentGameState).asList()
        if foodList:
            closestFood = min([self.getDistance(newPos, food) for food in foodList])
        else:
            closestFood = 0

        return -2 * closestFood + ghost_distance - 10 * len(foodList) + closest_capsule

    def expectimax(self, agent, depth, gameState):
        start = time.time()
        if gameState.isOver() or depth >= 3:  # return the utility in case the defined depth is reached or the game is won/lost.
            print(time.time() - start)
            return self.betterEvaluationFunction(gameState)
            # return self.evaluate(gameState, action)
        if agent == 0 or agent == 2:  # maximizing for pacman
            maxEval = float("-inf")
            for newState in gameState.getLegalActions(agent):
                eval = self.expectimax(agent + 1, depth, gameState.generateSuccessor(agent, newState))
                maxEval = max(maxEval, eval)
            print(time.time() - start)
            return maxEval
        else:  # performing expectimax action for ghosts/chance nodes.
            nextAgent = agent + 1  # calculate the next agent and increase depth accordingly.
            if gameState.getNumAgents() == nextAgent:
                nextAgent = 0
            if nextAgent == 0:
                depth += 1
            maxEval = 0
            for newState in gameState.getLegalActions(agent):
                eval = self.expectimax(nextAgent, depth, gameState.generateSuccessor(agent, newState))
                maxEval = maxEval + eval
            # print(maxEval)
            # print(float(len(gameState.getLegalActions(agent))))
            print(time.time() - start)
            return maxEval / float(len(gameState.getLegalActions(agent)))


    def expectiMax(self, gameState, agent, depth):
        result = []

        if not gameState.getLegalActions(agent):
            return self.betterEvaluationFunction(gameState), 0

        if depth == 1:
            return self.betterEvaluationFunction(gameState), 0

        if agent == gameState.getNumAgents() - 1:
            depth += 1
            nextAgent = self.index

        else:
            nextAgent = agent + 1

        for action in gameState.getLegalActions(agent):
            if not result:
                nextValue = self.expectiMax(gameState.generateSuccessor(agent, action), nextAgent, depth)

                if (agent != self.index):
                    result.append((1.0 / len(gameState.getLegalActions(agent))) * nextValue[0])
                    result.append(action)
                else:
                    result.append(nextValue[0])
                    result.append(action)
            else:

                previousValue = result[0]
                nextValue = self.expectiMax(gameState.generateSuccessor(agent, action), nextAgent, depth)

                if agent == self.index:
                    if nextValue[0] > previousValue:
                        result[0] = nextValue[0]
                        result[1] = action

                else:
                    result[0] = result[0] + (1.0 / len(gameState.getLegalActions(agent))) * nextValue[0]
                    result[1] = action
        return result


    def getDistance(self, myPos, food):
        return self.getMazeDistance(myPos, food) + abs(self.favoredY - food[1])


class BottomAgent(ExpectimaxAgent):
    def registerInitialState(self, gameState):
        ExpectimaxAgent.registerInitialState(self, gameState)
        self.favoredY = 0.0


class TopAgent(ExpectimaxAgent):
    def registerInitialState(self, gameState):
        ExpectimaxAgent.registerInitialState(self, gameState)
        self.favoredY = gameState.data.layout.height

