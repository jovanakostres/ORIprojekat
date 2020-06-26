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
from pacman_project2.pacman_project2 import util
from pacman_project2.pacman_project2.captureAgents import CaptureAgent
import random, time
from pacman_project2.pacman_project2.game import Directions, Actions
from pacman_project2.pacman_project2.util import nearestPoint
import pacman_project2.pacman_project2.game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAgent', second = 'DummyAgent'):
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
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

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

    return random.choice(bestActions)

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


class DummyAgent(ReflexCaptureAgent):
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
                    appendFood.append((foodX,foodY))
            if len(appendFood) == 0:
                appendFood = foodList
            if min([self.getMazeDistance(myPos, food) for food in appendFood]) is not None:
                features['distanceToFood'] = float(min([self.getMazeDistance(myPos, food) for food in appendFood]))/(walls.width * walls.height)

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
                    #moves away from chasers
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

    # returns the weights with priority set
    def getWeights(self, gameState, action):
        return {'invaderDistance': -10, 'distanceToFood':-1, 'attack-ghosts' : -100, 'escape': -30, 'stop': -5}