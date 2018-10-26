import gym
import numpy as np
import gym.spaces

def table_Q():
    env = gym.make("FrozenLake-v0")

    lr = 0.8
    gamma = 0.95

    Q = np.zeros([env.observation_space.n,env.action_space.n])

    episode = 2000
    epsilon = 0.1

    rList = []

    for i in range(episode):
        s = env.reset()
        done = False
        rAll = 0

        for j in range(99):
            '''
            if np.random.uniform() > epsilon:
                action = np.argmax(Q[s,:])
            else:
                action = np.random.choice(4)
            '''

            action = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
            #print(action)
            s1, r, d, _ = env.step(action)

            Q[s,action] = Q[s,action] + lr * (r + gamma*np.max(Q[s1,:]) - Q[s,action])

            rAll += r
            s = s1

            if done == True:
                break

        rList.append(rAll)

    print("Score over time:" + str(np.sum(rList)/episode))

    print(Q)

if __name__ == "__main__":
    table_Q()
