from gridworld.iter.policy import PolicyIter

env = PolicyIter([
    [0, 0, 0, -1],
    [0, -1, 0, 0],
    [0, -1, 10, 0]
])

if __name__ == '__main__':
    env.solve()
    print(env.policy)
