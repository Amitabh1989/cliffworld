import gym, cv2
import numpy as np
# from cv_show import initialize_frame, put_agent
import cv_show


cliffenv = gym.make("CliffWalking-v0", render_mode="ansi")

done = False
state = cliffenv.reset()
frame = cv_show.initialize_frame()
i = 1
# while not done:
#     # print("*"*30)
#     # print("Loop %s" %i)
#     # print("*" * 30)
#     # print(cliffenv.render())
#     frame2 = cv_show.put_agent(frame.copy(), state)
#     cv2.imshow("Cliff Walking", frame2)
#     cv2.waitKey(250)
#     action = int(np.random.randint(low=0, high=4, size=1))
#     # print("State : ", state, "----> Action : ", ["UP", "RIGHT", "DOWN", "LEFT"][action])
#     # step returns :
#     # Observation  : Object
#     # Reward       : Float => value returned as a result of an action taken
#     # Terminated   : Boolean : If a terminal stage has reached
#     # Truncated    : To check if an agent is going off the boundary of MDP
#     # Info         : Dictionary
#     state, reward, done, truncated, info = cliffenv.step(action)
#     # print("Observation o/r State : ", state)
#     # print("Reward                : ", reward)
#     # print("Terminated            : ", done)
#     # print("Truncated             : ", truncated)
#     # print("Info                  : ", info)
#     # print("\n" * 2)
#     # i += 1

while not done:
    frame2 = cv_show.put_agent(frame.copy(), state)
    cv2.imshow("Cliff Walking", frame2)
    cv2.waitKey(250)
    action = int(np.random.randint(low=0, high=4, size=1))
    state, reward, done, truncated, info = cliffenv.step(action)

# cliffenv.reset()
cliffenv.close()