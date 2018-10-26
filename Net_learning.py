import gym
import gym.spaces
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class tf_model():
    def __init__(self):
        tf.reset_default_graph()
        self.inputs = tf.placeholder(shape = [1,16], dtype=tf.float32)
        self.W = tf.Variable(tf.random_uniform([16,4],0,0.01))
        self.Qout = tf.matmul(self.inputs, self.W)
        self.predict = tf.argmax(self.Qout,1)

        self.next_Q = tf.placeholder(shape=[1,4],dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.Qout - self.next_Q))
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        self.update = self.trainer.minimize(self.loss)

        self.init = tf.global_variables_initializer()


def Net_learing():

    env = gym.make('FrozenLake-v0')

    model = tf_model()

    gamma = 0.95

    num_episode = 2000
    epsilon = 0.1
    rList = []
    jList = []

    with tf.Session() as sess:
        sess.run(model.init)

        for i in range(num_episode):
            s = env.reset()
            done = False
            rAll = 0
            j = 0

            while j < 99:
                j = j + 1

                action , Qout = sess.run([model.predict, model.Qout], feed_dict={model.inputs: np.identity(16)[s:s+1]})
                if np.random.rand(1) < epsilon:
                    action[0] = env.action_space.sample()

                s1,r,done,_ = env.step(action[0])

                rAll += r
                Q1 = sess.run(model.Qout, feed_dict={model.inputs:np.identity(16)[s1:s1+1]})

                Qmax = np.max(Q1)
                targetQ = Qout
                targetQ[0,action[0]] = r + gamma*Qmax

                _, W = sess.run([model.update, model.W], feed_dict={model.inputs:np.identity(16)[s:s+1], model.next_Q: targetQ})

                s = s1
                if done is True:
                    epsilon = 1. / ((i / 50) + 10)
                    break

            jList.append(j)
            rList.append(rAll)
    print("Percent of succesful episodes: " + str(sum(rList) / num_episode) + "%")

    #print(rList)
    plt.plot(rList)

if __name__ == "__main__":
    Net_learing()
