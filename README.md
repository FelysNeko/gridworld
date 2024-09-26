# GridWorld

Handwritten implementation of some basic reinforcement learning algorithms in a gridworld.

## Usage

Make sure `numpy` is installed, and then run:

```shell
python3 demo.py
```
The [implementation](gridworld) is more interesting than actually using it, since that is where maths comes to play. In case you forgot, here is the bellman optimality equation:

$$
V^*(s) = \max_a \sum_{s'} P(s'|s, a) \left[ R(s, a, s') + \gamma V^*(s') \right]
$$

## License

Distributed under the terms of the [MIT License](LICENSE).

## Copyright

© All rights reserved by FelysNeko
