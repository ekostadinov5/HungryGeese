
import tensorflow as tf


# DQN & DDQN
class Model(tf.keras.Model):

    def __init__(self, num_actions):
        super(Model, self).__init__()
        self.layer1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3)
        self.layer2 = tf.keras.layers.BatchNormalization()
        self.layer3 = tf.keras.layers.Conv2D(filters=16, kernel_size=3)
        self.layer4 = tf.keras.layers.BatchNormalization()
        self.layer5 = tf.keras.layers.Conv2D(filters=16, kernel_size=3)
        self.layer6 = tf.keras.layers.BatchNormalization()
        self.layer7 = tf.keras.layers.Conv2D(filters=16, kernel_size=3)
        self.layer8 = tf.keras.layers.BatchNormalization()
        self.layer9 = tf.keras.layers.Conv2D(filters=16, kernel_size=3)
        self.layer10 = tf.keras.layers.BatchNormalization()
        self.layer11 = tf.keras.layers.Flatten()
        self.layer12 = tf.keras.layers.Dense(32)
        self.layer13 = tf.keras.layers.Dense(num_actions)

    def call(self, x, **kwargs):
        """Forward pass"""
        x = self.layer1(x)
        x = self.layer2(x)
        x = tf.keras.activations.relu(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = tf.keras.activations.relu(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = tf.keras.activations.relu(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = tf.keras.activations.relu(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = tf.keras.activations.relu(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = tf.keras.activations.relu(x)
        x = self.layer13(x)

        return x


# PPO
class Actor(tf.keras.Model):

    def __init__(self, num_actions):
        super(Actor, self).__init__()
        self.layer1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3)
        self.layer2 = tf.keras.layers.BatchNormalization()
        self.layer3 = tf.keras.layers.Conv2D(filters=16, kernel_size=3)
        self.layer4 = tf.keras.layers.BatchNormalization()
        self.layer5 = tf.keras.layers.Conv2D(filters=16, kernel_size=3)
        self.layer6 = tf.keras.layers.BatchNormalization()
        self.layer7 = tf.keras.layers.Conv2D(filters=16, kernel_size=3)
        self.layer8 = tf.keras.layers.BatchNormalization()
        self.layer9 = tf.keras.layers.Conv2D(filters=16, kernel_size=3)
        self.layer10 = tf.keras.layers.BatchNormalization()
        self.layer11 = tf.keras.layers.Flatten()
        self.layer12 = tf.keras.layers.Dense(32)
        self.layer13 = tf.keras.layers.Dense(num_actions, activation='softmax')

    def call(self, x, **kwargs):
        """Forward pass"""
        x = self.layer1(x)
        x = self.layer2(x)
        x = tf.keras.activations.relu(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = tf.keras.activations.relu(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = tf.keras.activations.relu(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = tf.keras.activations.relu(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = tf.keras.activations.relu(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = tf.keras.activations.relu(x)
        x = self.layer13(x)

        return x + 1e-7


class Critic(tf.keras.Model):

    def __init__(self):
        super(Critic, self).__init__()
        self.layer1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3)
        self.layer2 = tf.keras.layers.BatchNormalization()
        self.layer3 = tf.keras.layers.Conv2D(filters=16, kernel_size=3)
        self.layer4 = tf.keras.layers.BatchNormalization()
        self.layer5 = tf.keras.layers.Conv2D(filters=16, kernel_size=3)
        self.layer6 = tf.keras.layers.BatchNormalization()
        self.layer7 = tf.keras.layers.Conv2D(filters=16, kernel_size=3)
        self.layer8 = tf.keras.layers.BatchNormalization()
        self.layer9 = tf.keras.layers.Conv2D(filters=16, kernel_size=3)
        self.layer10 = tf.keras.layers.BatchNormalization()
        self.layer11 = tf.keras.layers.Flatten()
        self.layer12 = tf.keras.layers.Dense(32)
        self.layer13 = tf.keras.layers.Dense(1, activation='linear')

    def call(self, x, **kwargs):
        """Forward pass"""
        x = self.layer1(x)
        x = self.layer2(x)
        x = tf.keras.activations.relu(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = tf.keras.activations.relu(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = tf.keras.activations.relu(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = tf.keras.activations.relu(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = tf.keras.activations.relu(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = tf.keras.activations.relu(x)
        x = self.layer13(x)

        return x
