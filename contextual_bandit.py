import numpy as np
import tensorflow as tf


class con_bandit():
    def __init__(self):
        self.bandits = np.array([[0.2, 0, -0.0, -5],[0.1,-5,1,0.25],[-5,5,5,5]])
        self.state = 0
        self.s_size = self.bandits.shape[0]
        self.a_size = self.bandits.shape[1]

    def get_bandit(self):
        self.state = np.random.randint(0, self.s_size)
        return self.state

    def pull_arm(self, action):
        bandit = self.bandits[self.state, action]
        if np.random.randn(1) > bandit:
            return 1
        else:
            return -1


class tf_model():
    def __init__(self, s_size, a_size):
        self.state = tf.placeholder(shape=[1], dtype=tf.int32)
        self.one_hot = tf.one_hot(self.state, depth=s_size)

        self.Weight = tf.get_variable('w', shape=[s_size,a_size], initializer=tf.contrib.layers.xavier_initializer())
        #self.B = tf.Variable(tf.zeros([a_size]))
        self.output = tf.nn.sigmoid(tf.matmul(self.one_hot, self.Weight))
        print(self.output)
        self.output = tf.reshape(self.output,[-1])
        self.choose_a = tf.argmax(self.output,0)

        self.reward = tf.placeholder(shape=[1], dtype=tf.float32)
        self.action = tf.placeholder(shape=[1], dtype=tf.int32)
        self.responsible_weight = tf.slice(self.output, self.action, [1])
        self.loss = -(tf.log(self.responsible_weight)*self.reward)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
        self.update = optimizer.minimize(self.loss)
        self.init = tf.global_variables_initializer()


def run():
    tf.reset_default_graph()

    contextual_ban = con_bandit()
    model = tf_model(contextual_ban.s_size, contextual_ban.a_size)

    num_episode = 10000
    epsilon = 0.1
    total_reward = np.zeros([contextual_ban.s_size, contextual_ban.a_size])

    with tf.Session() as sess:
        sess.run(model.init)

        for i in range(num_episode):
            state = contextual_ban.get_bandit()

            if np.random.rand(1) < epsilon:
                action = np.random.choice(contextual_ban.a_size)

            else:
                action = sess.run(model.choose_a, feed_dict={model.state: [state]})


            reward = contextual_ban.pull_arm(action)

            _, weight = sess.run([model.update, model.Weight], feed_dict={model.state: [state], model.reward:[reward], model.action:[action]})

            total_reward[state,action] += reward

            if i % 1000 == 0:
                print("average reward: ", np.mean(total_reward, axis=1))
                print(weight)
        print("best action:", total_reward)

if __name__ == "__main__":
    run()