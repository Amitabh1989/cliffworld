import gym
import numpy as np
import pickle as pkl

env = gym.make("CliffWalking-v0")
q_table = np.zeros(shape=(48, 4)) # Coz the table has 48 cells and each cell can have 4 actions / movements

def policy(state, explore=0.0):
    action = int(np.argmax(q_table[state]))
    # With 1-epsilon it takes an optimal action
    # With epsilon probability, it takes in random action
    if np.random.random() <= explore:
        action = int(np.random.randint(low=0, high=1, size=1))
    return action

# Parameters
EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
NUM_EPISODES = 500


for episode in range(NUM_EPISODES):
    done = False
    total_reward = 0
    episode_lenght = 0

    state = env.reset()
    state = int(state[0]) if isinstance(state, tuple) else int(state)
    action = policy(state, EPSILON)

    while not done:
        next_state, reward, done, truncated, info = env.step(action)
        next_action = policy(next_state, EPSILON)

        # Perform our SARSA update

        q_table[state][action] += ALPHA * (reward + GAMMA * q_table[next_state][next_action] - q_table[state][action])
        state = next_state
        action = next_action
        total_reward += reward
        episode_lenght += 1

    print("EPISODE : ", episode ,"   Episode Length : ", episode_lenght, "   Total Reward : ", total_reward)

env.close()
pkl.dump(q_table, open("sarsa_q_table.pkl", "wb"))
print("Training complete, q_table saved !!")