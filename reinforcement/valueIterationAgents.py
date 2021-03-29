# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for _ in range(self.iterations):
            old_values = self.values.copy()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    continue
                value = -float('inf')
                for action in self.mdp.getPossibleActions(state):
                    values = 0
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        values += prob * (self.mdp.getReward(state, action, nextState) + self.discount * old_values[nextState])
                    value = max(values, value)
                self.values[state] = value

                


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        qvalue = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            qvalue += prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.values[nextState])
        return qvalue
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        max_value, best_action = -float('inf'), None
        for action in self.mdp.getPossibleActions(state):
            qvalue = self.getQValue(state, action)
            if qvalue > max_value:
                max_value, best_action = qvalue, action
        return best_action
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            states = self.mdp.getStates()
            i %= len(states)
            if self.mdp.isTerminal(states[i]):
                continue
            value = -float('inf')
            for action in self.mdp.getPossibleActions(states[i]):
                values = 0
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(states[i], action):
                    values += prob * (self.mdp.getReward(states[i], action, nextState) + self.discount * self.values[nextState])
                value = max(values, value)
            self.values[states[i]] = value

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

        

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        """get the predecessors"""
        """
        predecessors = {}
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            for action in self.mdp.getPossibleActions(state):
                for nextState, _ in self.mdp.getTransitionStatesAndProbs(state, action):
                    if nextState in predecessors:
                        predecessors[nextState].add(state)
                    else:
                        predecessors[nextState] = {state}

        queue = util.PriorityQueue()
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            maxq = max(self.getQValue(state, action) for action in self.mdp.getPossibleActions(state))
            diff = abs(self.values[state] - maxq)
            queue.update((state, maxq), -diff)
        for _ in range(self.iterations):
            if queue.isEmpty():
                return
            state, newValue = queue.pop()
            self.values[state] = newValue
            for p in predecessors[state]:
                if self.mdp.isTerminal(p):
                    continue
                q = max(self.getQValue(p, action) for action in self.mdp.getPossibleActions(p))
                d = abs(self.values[p] - q)
                if d > self.theta:
                    queue.update((p, q), -d)
                    """
                    

        queue = util.PriorityQueue()
        predecessors = {}

        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            for action in self.mdp.getPossibleActions(state):
                for nextState, _ in self.mdp.getTransitionStatesAndProbs(state, action):
                    if nextState in predecessors:
                        predecessors[nextState].add(state)
                    else:
                        predecessors[nextState] = {state}


        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            diff = abs(self.values[state] - max(self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)))
            queue.update(state, -diff)

        for _ in range(self.iterations):
            if queue.isEmpty():
                return
            state = queue.pop()
            if not self.mdp.isTerminal(state):
                self.values[state] = max(self.getQValue(state, action) for action in self.mdp.getPossibleActions(state))
            for p in predecessors[state]:
                if self.mdp.isTerminal(p):
                    continue
                diff = abs(self.values[p] - max(self.getQValue(p, action) for action in self.mdp.getPossibleActions(p)))
                if diff > self.theta:
                    queue.update(p, -diff)


