import gym
import numpy as np
import pickle as pkl


env = gym.make("CliffWalking-v0")
env.reset()

q_table = np.zeros(shape=(48, 4))

# Parameters
EPISODES = 500
EPSILON = 0.1
GAMMA = 0.9
ALPHA = 0.1


def policy(state, explore=0.0):
    action = int(np.argmax(q_table[state]))

    if np.random.random() <= explore:
        action = int(np.random.randint(low=0, high=1, size=1))

    return action

for episode in range(EPISODES):
    done = False
    state = env.reset()
    state = int(state[0]) if isinstance(state, tuple) else int(state)
    total_reward = 0
    episode_length = 0


    while not done:
        action = policy(state, EPSILON)
        next_state, reward, done, truncated, info = env.step(action)
        next_action = policy(next_state)
        total_reward += reward
        q_table[state][action] += ALPHA * (reward + GAMMA * q_table[next_state][next_action] - q_table[state][action])
        state = next_state
        episode_length += 1
    print("EPISODE : ", episode ,"   Episode Length : ", episode_length, "   Total Reward : ", total_reward)

env.close()
pkl.dump(q_table, open('q_learning_table.pkl', 'wb'))


