
import numpy as np
from kaggle_environments import evaluate


def eval():
    scores = evaluate('hungry_geese', ['greedy', 'submission.py', 'greedy', 'greedy'], num_episodes=100)
    scoreboard = [0, 0, 0, 0]
    for score in scores:
        winner = np.argmax(score)
        scoreboard[winner] += 1
    print(scores)
    print(scoreboard)
    print()
