
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from models import Model
from models import Actor, Critic


# DQN & DDQN
class QAgent:

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


class DQNAgent(QAgent):

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


class DDQNAgent(QAgent):

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


# (Truly) PPO
class PPOAgent():

    def __init__(self, rows, columns, num_actions, l_rate=1e-4, gamma=0.99, lam=0.95, policy_kl_range=0.0008,
                 policy_params=20, value_clip=1.0, loss_coefficient=1.0, entropy_coefficient=0.05):
        self.rows = rows
        self.columns = columns
        self.num_actions = num_actions

        self.actor = Actor(self.num_actions)
        self.critic = Critic()
        self.actor_old = Actor(self.num_actions)
        self.critic_old = Critic()

        self.optimizer = tf.keras.optimizers.Adam(l_rate)

        self.gamma = gamma
        self.lam = lam
        self.policy_kl_range = policy_kl_range
        self.policy_params = policy_params
        self.value_clip = value_clip
        self.loss_coefficient = loss_coefficient
        self.entropy_coefficient = entropy_coefficient

    @tf.function
    def fit(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            action_probabilities, values = self.actor(states), self.critic(states)
            old_action_probabilities, old_values = self.actor_old(states), self.critic_old(states)
            next_values = self.critic(next_states)
            loss = self._get_loss(action_probabilities, values, old_action_probabilities, old_values, next_values,
                                  actions, rewards, dones)
        grads = tape.gradient(loss, self.actor.trainable_variables + self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables + self.critic.trainable_variables))

    def select_action(self, state, training=False):
        state_in = tf.expand_dims(state, axis=0)
        probabilities = self.actor(state_in)

        if training:
            distribution = tfp.distributions.Categorical(probs=probabilities)
            action = distribution.sample()
            action = int(action[0])
        else:
            action = tf.argmax(probabilities[0]).numpy()

        return action

    def get_model(self):
        return self.actor

    def update_networks(self):
        self.actor_old.set_weights(self.actor.get_weights())
        self.critic_old.set_weights(self.critic.get_weights())

    def save_model_weights(self, actor_filename, critic_filename):
        self.actor.save_weights(actor_filename)
        self.critic.save_weights(critic_filename)

    def load_model_weights(self, actor_filename, critic_filename=None):
        self.actor(np.zeros((1, self.rows, self.columns, 1)))
        self.actor.load_weights(actor_filename)
        self.actor_old(np.zeros((1, self.rows, self.columns, 1)))
        self.actor_old.load_weights(actor_filename)

        if critic_filename is not None:
            self.critic(np.zeros((1, self.rows, self.columns, 1)))
            self.critic.load_weights(critic_filename)
            self.critic_old(np.zeros((1, self.rows, self.columns, 1)))
            self.critic_old.load_weights(critic_filename)

    def save_optimizer_weights(self, filename):
        np.save(filename, self.optimizer.get_weights())

    def load_optimizer_weights(self, filename):
        optimizer_weights = np.load(filename, allow_pickle=True)
        model_weights = self.actor.trainable_weights + self.critic.trainable_variables
        zero_grads = [tf.zeros_like(w) for w in model_weights]
        self.optimizer.apply_gradients(zip(zero_grads, model_weights))
        self.optimizer.set_weights(optimizer_weights)

    def _get_loss(self, action_probabilities, values, old_action_probabilities, old_values, next_values, actions,
                  rewards, dones):
        old_values = tf.stop_gradient(old_values)

        advantages = self._generalized_advantages_estimation(values, rewards, next_values, dones)
        returns = tf.stop_gradient(advantages + values)
        advantages = \
            tf.stop_gradient((advantages - tf.math.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-7))

        log_probabilities = self._log_probabilities(action_probabilities, actions)
        old_log_probabilities = tf.stop_gradient(self._log_probabilities(old_action_probabilities, actions))
        ratios = tf.math.exp(log_probabilities - old_log_probabilities)

        kl_divergence = self._kl_divergence(old_action_probabilities, action_probabilities)

        policy_gradient_loss = tf.where(
            tf.logical_and(kl_divergence >= self.policy_kl_range, ratios > 1),
            ratios * advantages - self.policy_params * kl_divergence,
            ratios * advantages
        )
        policy_gradient_loss = tf.math.reduce_mean(policy_gradient_loss)

        entropy = tf.math.reduce_mean(self._entropy(action_probabilities))

        clipped_values = old_values + tf.clip_by_value(values - old_values, -self.value_clip, self.value_clip)
        values_losses = tf.math.square(returns - values) * 0.5
        clipped_values_losses = tf.math.square(returns - clipped_values) * 0.5

        critic_loss = tf.math.reduce_mean(tf.math.maximum(values_losses, clipped_values_losses))
        loss = (critic_loss * self.loss_coefficient) - policy_gradient_loss - (entropy * self.entropy_coefficient)

        return loss

    def _generalized_advantages_estimation(self, values, rewards, next_values, dones):
        gae = 0
        advantages = []
        delta = rewards + (1.0 - dones) * self.gamma * next_values - values
        for i in reversed(range(len(rewards))):
            gae = delta[i] + (1.0 - dones[i]) * self.gamma * self.lam * gae
            advantages.insert(0, gae)

        return tf.stack(advantages)

    def _log_probabilities(self, action_probabilities, actions):
        distribution = tfp.distributions.Categorical(probs=action_probabilities)
        return tf.expand_dims(distribution.log_prob(actions), axis=1)

    def _kl_divergence(self, probabilities1, probabilities2):
        distribution1 = tfp.distributions.Categorical(probs=probabilities1)
        distribution2 = tfp.distributions.Categorical(probs=probabilities2)
        return tf.expand_dims(tfp.distributions.kl_divergence(distribution1, distribution2), axis=1)

    def _entropy(self, probabilities):
        distribution = tfp.distributions.Categorical(probs=probabilities)
        return distribution.entropy()
