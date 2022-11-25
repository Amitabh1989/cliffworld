import gym
import numpy as np


cliffenv = gym.make("CliffWalking-v0")

done = False
state = cliffenv.reset()

while not done:
    print(cliffenv.render(mode="ansi"))
    action = int(np.random.randint(low=0, high=4, size=1))
    print(state, "---->", action)
    state, reward, done, _ = cliffenv.step(action)

cliffenv.close()