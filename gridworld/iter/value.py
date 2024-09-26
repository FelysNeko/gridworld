from typing import List

import numpy as np

from gridworld.environ import GridWorld


class ValueIter(GridWorld):
    def __init__(self, reward: List[List[int]], gamma: float = 0.9):
        super().__init__(reward, gamma)

    def flush_v(self):
        for state in self.states():
            action_values = []
            for action in range(5):
                ns = self.move(state, action)
                r = self.rewards[ns]
                value = r + self.gamma * self.V[ns]
                action_values.append(value)
            self.V[state] = max(action_values)

    def dp_update_v(self, threshold: float = 0.001):
        while True:
            old_v = self.V.copy()
            self.flush_v()
            diff = abs(old_v - self.V)
            if diff.max() < threshold:
                break

    def argmax_policies(self):
        for state in self.states():
            action_values = {}
            for action, prob in enumerate(self.policy[state]):
                ns = self.move(state, action)
                r = self.rewards[ns]
                action_values[r + self.gamma * self.V[ns]] = action
            argmax = action_values[max(action_values)]
            new_policy = np.zeros(5)
            new_policy[argmax] = 1
            self.policy[state] = new_policy

    def solve(self):
        self.dp_update_v()
        self.argmax_policies()
