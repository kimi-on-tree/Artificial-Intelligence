# LAB1

## J0

name1: Mo Kong

email1: mkong@kth.se

name2: None

## J1



## J2

> **Q1:** Describe the state space, what specifies the initial state of a scenario, and the successor function of the KTH fishing derby game.

**The state space:** The current player's index; The player scores; The positions of the two hooks; The positions of the uncaught fishes; The score values associated with each fish index. **Initial state** of a scenario is decided by the observation files. The file offers the initial position of each player's hook, the initial position of fish and value, and each fish movement sequence. **The successor function** applies one of five actions, move the fish by the next observation, update catches and scores, and swich player.

> Q2: Describe the terminal states of the KTH fishing derby game. What utility should be assigned to a terminal state from the perspective of the green/MAX player?

The terminal states: 1. There's no fish left 2. The observation sequence is exhausted. For the green player, the utility of a terminal state is the final score difference(positive=win; negative=loss; zero=tie)

