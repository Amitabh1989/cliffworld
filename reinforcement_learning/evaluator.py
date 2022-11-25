import gym
import pickle as pkl
import cv2
import numpy as np
import cv_show
import time

env = gym.make("CliffWalking-v0")

q_table = pkl.load(open("sarsa_q_table.pkl", "rb"))

time.sleep(10)
def policy(state, explore=0.0):
    action = int(np.argmax(q_table[state]))
    # With 1-epsilon it takes an optimal action
    # With epsilon probability, it takes in random action
    if np.random.random() <= explore:
        action = int(np.random.randint(low=0, high=1, size=1))
    return action

# Parameters
NUM_EPISODES = 5

for episode in range(NUM_EPISODES):
    done = False
    frame = cv_show.initialize_frame()
    state = env.reset()
    state = int(state[0]) if isinstance(state, tuple) else int(state)
    total_reward = 0
    episode_length = 0

    while not done:
        frame2 = cv_show.put_agent(frame.copy(), state)
        cv2.imshow("Cliff Walking", frame2)
        cv2.waitKey(250)
        action = policy(state)
        state, reward, done, truncated, info = env.step(action)

        total_reward += reward
        episode_length += 1
    print("EPISODE : ", episode, "   Episode Length : ", episode_length, "   Total Reward : ", total_reward)

env.close()
