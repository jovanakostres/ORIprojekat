from pacman_project2.pacman_project2.captureAgents import CaptureAgent
from pacman_project2.pacman_project2 import util
from pacman_project2.pacman_project2.util import nearestPoint
import random

def createTeam(firstIndex, secondIndex, isRed, first = 'DAgent', second = 'OAgent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

class RefCaptureAgent(CaptureAgent):
    SEARCH_DEPTH = 5

    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 1.0}

    def evaluationFunction(self, currentGameState, action):
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

        distances_to_ghosts = 1
        proximity_to_ghosts = 0
        for ghost_state in successorGameState.getGhostPositions():
            distance = util.manhattanDistance(newPos, ghost_state)
            distances_to_ghosts += distance
            if distance <= 1:
                proximity_to_ghosts += 1

        return successorGameState.getScore() + (1 / float(min_food_distance)) - (
                1 / float(distances_to_ghosts)) - proximity_to_ghosts

        # return successorGameState.getScore()

    def chooseAction(self, gameState):

        actions = gameState.getLegalActions(self.index)



        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        # values = [self.expectimax(1, 0, gameState.generateSuccessor(0, a)) for a in actions]

        # values = [self.evaluate(gameState, a) for a in actions]
        #print(self.index)

        indexPlus = self.index + 1
        if (indexPlus == 4):
            indexPlus = 0

        opponents = self.getOpponents(gameState)

        # values = [self.expectiMax2(gameState, indexPlus, 0, a)[0] for a in actions]
        values = [self.expectinegamax(a, gameState, self.SEARCH_DEPTH, 1, retAction=True) for a in opponents]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)


        bestValue = float("-inf")
        bestActions = []

        for v in values:
            if v[0] >= bestValue :
                bestActions.append(v[1])

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

        bestAction = random.choice(bestActions)
        return bestAction

    def expectinegamax(self, opponent, state, depth, sign, retAction=False):
        """
        Negamax variation of expectimax.
        """
        if sign == 1:
          agent = self.index
        else:
          agent = opponent



        bestAction = None
        if len(state.getLegalActions(agent)) == 0 or depth == 0:
          bestVal = sign * self.evaluateState(state)
        else:
          actions = state.getLegalActions(agent)
          actions.remove('Stop')
          bestVal = float("-inf") if agent == self.index else 0
          for action in actions:
            successor = state.generateSuccessor(agent, action)
            value = -self.expectinegamax(opponent, successor, depth - 1, -sign)
            if agent == self.index and value > bestVal:
              bestVal, bestAction = value, action
            elif agent == opponent:
              bestVal += value/len(actions)

        if agent == self.index and retAction:
          return bestVal, bestAction
        else:
          return bestVal

    def getOpponentDistances(self, gameState):
        """
        Return the IDs of and distances to opponents, relative to this agent.
        """
        return [(o, self.distancer.getDistance(
            gameState.getAgentPosition(self.index),
            gameState.getAgentPosition(o)))
                for o in self.getOpponents(gameState)]

def scoreEvaluationFunction(currentGameState):
    return currentGameState.getScore()

class DAgent(RefCaptureAgent): # agent koji brani svoj deo od protivnickih Pacman-a
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        # self.favoredY = 0.0
        CaptureAgent.registerInitialState(self, gameState)

    def evaluateState(self, gameState):
        myPosition = gameState.getAgentPosition(self.index)
        food = self.getFood(gameState).asList()

        targetFood = None
        maxDist = 0

        opponentDistances = self.getOpponentDistances(gameState)
        opponentDistance = min([dist for id, dist in opponentDistances])

        if not food or gameState.getAgentState(self.index).numCarrying > self.getScore(gameState) > 0:
            return 20 * self.getScore(gameState) \
                   - self.distancer.getDistance(myPosition, gameState.getInitialAgentPosition(self.index)) \
                   + opponentDistance

        for f in food:
            d = min([self.distancer.getDistance(gameState.getAgentPosition(o), f)
                     for o in self.getOpponents(gameState)])
            if d > maxDist:
                targetFood = f
                maxDist = d
        if targetFood:
            foodDist = self.distancer.getDistance(myPosition, targetFood)
        else:
            foodDist = 0

        distanceFromStart = abs(myPosition[0] - gameState.getInitialAgentPosition(self.index)[0])
        if not len(food):
            distanceFromStart *= -1

        return 2 * self.getScore(gameState) \
               - 100 * len(food) \
               - 2 * foodDist \
               + opponentDistance \
               + distanceFromStart


class OAgent(RefCaptureAgent):
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        # self.favoredY = 0.0
        CaptureAgent.registerInitialState(self, gameState)

    def evaluateState(self, gameState):

        myPosition = gameState.getAgentPosition(self.index)
        if gameState.isRed(myPosition) != gameState.isOnRedTeam(self.index):
            return -1000000
        score = 0
        pacmanState = []
        for opponent in self.getOpponents(gameState):
            opPosition = gameState.getAgentPosition(opponent)
            pacmanState.append(gameState.isRed(opPosition)!=gameState.isOnRedTeam(opponent))

        opponentDistances = self.getOpponentDistances(gameState)

        for isPacman, (id, distance) in zip(pacmanState, opponentDistances):
            if isPacman:
                score -= 100000
                score -= 5 * distance
            elif not any(pacmanState):
                score -= distance

        return score