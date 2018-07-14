import gym
import gym.spaces
import numpy as np

if __name__ == "__main__":
    env = gym.make('FrozenLake-v0')

    #initialize the table
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    #learing parameters
    lr = .8
    y = .95
    num_episodes = 2000

    rList = []

    for i in range(num_episodes):
        s = env.reset()
        rALL = 0
        d = False
        j = 0
        while j < 99:
            j = j + 1
            #select an action
            a = np.argmax(Q[s,:] + np.random.randn(1, env.action_space.n)*(1./(i+1)))

            #get new state and reward
            s1,r,d,_ = env.step(a)
            #Update Q table
            Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])

            rALL = rALL + r
            s = s1
            if d == True:
                break

        rList.append(rALL)

    print("Score over time:" + str(sum(rList)/num_episodes))
    print("Final Q")
    print(Q)