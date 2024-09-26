from gridworld.iter import ValueIter, PolicyIter

grid = [
    [0, 0, 0, -10],
    [0, -10, 0, 0],
    [0, -1, 10, 0]
]

if __name__ == '__main__':
    ValueIter(grid).render()
    PolicyIter(grid).render()
