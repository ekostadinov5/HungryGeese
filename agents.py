
import tensorflow as tf
import numpy as np
from models import Model


class Agent:

    def __init__(self, rows, columns, num_actions, l_rate=1e-4, gamma=0.99):
        self.rows = rows
        self.columns = columns
        self.num_actions = num_actions

        self.policy_nn = Model(self.num_actions)
        self.target_nn = Model(self.num_actions)

        self.optimizer = tf.keras.optimizers.Adam(l_rate)
        self.mse = tf.keras.losses.MeanSquaredError()

        self.gamma = gamma

    def select_action(self, state):
        state_in = tf.expand_dims(state, axis=0)
        return tf.argmax(self.policy_nn(state_in)[0]).numpy()

    def select_epsilon_greedy_action(self, state, epsilon):
        if tf.random.uniform((1,)) < epsilon:
            return np.random.choice([0, 1, 2])
        else:
            return self.select_action(state)

    def get_model(self):
        return self.policy_nn

    def save_model_weights(self, filename):
        self.policy_nn.save_weights(filename)

    def load_model_weights(self, filename):
        self.policy_nn(np.zeros((1, self.rows, self.columns, 1)))
        self.policy_nn.load_weights(filename)

        self.target_nn(np.zeros((1, self.rows, self.columns, 1)))
        self.target_nn.load_weights(filename)

    def save_optimizer_weights(self, filename):
        np.save(filename, self.optimizer.get_weights())

    def load_optimizer_weights(self, filename):
        optimizer_weights = np.load(filename, allow_pickle=True)
        model_weights = self.policy_nn.trainable_weights
        zero_grads = [tf.zeros_like(w) for w in model_weights]
        self.optimizer.apply_gradients(zip(zero_grads, model_weights))
        self.optimizer.set_weights(optimizer_weights)

    def update_target_network(self):
        self.target_nn(np.zeros((1, self.rows, self.columns, 1)))
        self.target_nn.set_weights(self.policy_nn.get_weights())


class DQNAgent(Agent):

    @tf.function
    def fit(self, states, actions, rewards, next_states, dones):
        next_qs = self.target_nn(next_states)
        max_next_qs = tf.reduce_max(next_qs, axis=-1)
        targets = rewards + (1 - dones) * max_next_qs * self.gamma
        with tf.GradientTape() as tape:
            qs = self.policy_nn(states)
            action_masks = tf.one_hot(actions, self.num_actions)
            masked_qs = tf.reduce_sum(action_masks * qs, axis=-1)
            loss = self.mse(targets, masked_qs)
        grads = tape.gradient(loss, self.policy_nn.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy_nn.trainable_variables))

        return loss


class DDQNAgent(Agent):

    @tf.function
    def fit(self, states, actions, rewards, next_states, dones):
        policy_next_qs = self.policy_nn(next_states)
        max_policy_next_qs = tf.expand_dims(tf.argmax(policy_next_qs, axis=-1), axis=1)
        next_qs = self.target_nn(next_states)
        max_next_qs = tf.gather_nd(next_qs, indices=max_policy_next_qs, batch_dims=1)
        targets = rewards + (1. - dones) * max_next_qs * self.gamma
        with tf.GradientTape() as tape:
            qs = self.policy_nn(states)
            action_masks = tf.one_hot(actions, self.num_actions)
            masked_qs = tf.reduce_sum(action_masks * qs, axis=-1)
            loss = self.mse(targets, masked_qs)
        grads = tape.gradient(loss, self.policy_nn.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy_nn.trainable_variables))

        return loss
