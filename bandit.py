import numpy as np
import tensorflow as tf


bandits = [0.2, 0, -0.2, -5]
num_bandits = len(bandits)


def pull_bandit(action):
    r = np.random.randn(1)
    if r > bandits[action]:
        return 1
    else:
        return -1


class tf_model():

    def __init__(self):
        tf.reset_default_graph()
        self.W = tf.Variable(tf.ones([num_bandits]))
        self.choose_a = tf.argmax(self.W,0)

        self.reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
        self.responsible_weight = tf.slice(self.W, self.action_holder, [1])
        self.loss = - tf.log(self.responsible_weight)*self.reward_holder
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        self.update = self.trainer.minimize(self.loss)
        self.init = tf.global_variables_initializer()


def run():

    model = tf_model()

    num_episode = 2000
    epsilon = 0.1
    total_reward = np.zeros([len(bandits)])

    with tf.Session() as sess:
        sess.run(model.init)

        for i in range(num_episode):
            if np.random.uniform() > epsilon:
                action = sess.run(model.choose_a)
            else:
                action = np.random.choice(len(bandits))

            reward = pull_bandit(action)

            _, resp, weights = sess.run([model.update, model.responsible_weight, model.W], feed_dict={model.reward_holder: [reward], model.action_holder: [action]})

            total_reward[action] += reward

            if i % 50 == 0:
                print("Running reward for all bandits is", str(total_reward))
        print(weights)


if __name__ == "__main__":
    run()