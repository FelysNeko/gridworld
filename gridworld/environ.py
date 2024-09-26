from typing import List, Tuple, Generator
import numpy as np

State = Tuple[int, int]


class GridWorld:
    def __init__(self, reward: List[List[int]], gamma: int = 0.9, wall: int = -10):
        self.rewards = np.array(reward)
        self.gamma = gamma
        self.wall = wall
        self.actions = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))
        self.height = self.rewards.shape[0]
        self.width = self.rewards.shape[1]
        self.V = np.zeros((self.height, self.width))
        self.policy = np.full((self.height, self.width, 5), .2)

    def move(self, state: State, action: int) -> State:
        move = self.actions[action]
        nx, ny = state[0] + move[0], state[1] + move[1]
        if 0 <= nx < self.height and 0 <= ny < self.width:
            return nx, ny
        else:
            return state

    def states(self) -> Generator[State, None, None]:
        for state in np.ndindex(self.rewards.shape):
            yield state

