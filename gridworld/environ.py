from abc import ABC, abstractmethod
from typing import List, Generator, Tuple

import numpy as np


class GridWorld(ABC):
    def __init__(self, reward: List[List[int]], gamma: float = 0.9) -> None:
        self.rewards = np.array(reward)
        self.gamma = gamma
        self.actions = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))
        self.height = self.rewards.shape[0]
        self.width = self.rewards.shape[1]
        self.V = np.zeros((self.height, self.width))
        self.policy = np.full((self.height, self.width, 5), .2)
        self.solve()

    def move(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        move = self.actions[action]
        ny, nx = state[0] + move[0], state[1] + move[1]
        if 0 <= ny < self.height and 0 <= nx < self.width:
            return ny, nx
        else:
            return state

    def states(self) -> Generator[Tuple[int, int], None, None]:
        for state in np.ndindex(self.rewards.shape):
            yield state

    def render(self) -> None:
        actions = ['↑', '↓', '←', '→', '*']
        policy = self.policy.argmax(axis=2)
        cvt = np.vectorize(lambda x: actions[x])
        print(cvt(policy))

    @abstractmethod
    def solve(self) -> None:
        pass
