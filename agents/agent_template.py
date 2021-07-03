
import io
import base64
import numpy as np
import tensorflow as tf
from kaggle_environments import make


def preprocess_state(obs_dict, prev_direction=0):
    index = obs_dict['index']
    goose = obs_dict['geese'][index]
    enemy_geese = obs_dict['geese'][:]
    del enemy_geese[index]
    food = obs_dict['food']

    rows, columns = 7, 11

    if not goose:
        return np.zeros((rows + 4, columns, 1))

    state = np.zeros((rows, columns))

    for i in range(len(goose)):
        row, column = get_position(goose[i])
        state[row, column] = 1 if i == 0 else -1 if i == len(goose) - 1 else -2

    for enemy_goose in enemy_geese:
        for i in range(len(enemy_goose)):
            row, column = get_position(enemy_goose[i])
            state[row, column] = -5 if i == 0 else -1 if i == len(enemy_goose) - 1 else -2

    for f in food:
        row, column = get_position(f)
        state[row, column] = 5

    head_row, head_column = get_position(goose[0])
    state = center_goose(state, head_row, head_column)

    extended_state = np.zeros((rows + 4, columns))

    first_two_rows = state[:2].copy()
    last_two_rows = state[-2:].copy()
    extended_state[:2] = last_two_rows
    extended_state[2:-2] = state
    extended_state[-2:] = first_two_rows
    extended_state = np.rot90(extended_state, prev_direction)

    return extended_state.reshape((rows + 4, columns, 1))


def get_position(x):
    return int(x / 11), x % 11


def center_goose(state, head_row, head_column):
    center_row, center_column = 3, 5
    diff_row, diff_column = head_row - center_row, head_column - center_column
    if diff_row > 0:
        temp = state[:diff_row].copy()
        state[:-diff_row] = state[diff_row:]
        state[-diff_row:] = temp
    elif diff_row < 0:
        temp = state[diff_row:].copy()
        state[-diff_row:] = state[:diff_row]
        state[:-diff_row] = temp
    if diff_column > 0:
        temp = state[:, :diff_column].copy()
        state[:, :-diff_column] = state[:, diff_column:]
        state[:, -diff_column:] = temp
    elif diff_column < 0:
        temp = state[:, diff_column:].copy()
        state[:, -diff_column:] = state[:, :diff_column]
        state[:, :-diff_column] = temp

    return state


def get_direction(prev_direction, action):
    new_direction = prev_direction + (0 if action == 0 else -1 if action == 1 else 1)
    if new_direction < 0:
        return 3
    if new_direction > 3:
        return 0
    return new_direction


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

        # DQN & DDQN
        self.layer13 = tf.keras.layers.Dense(num_actions)

        # PPO
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


class Agent:

    def __init__(self, rows, columns, num_actions):
        self.rows = rows
        self.columns = columns
        self.num_actions = num_actions
        self.model = Model(self.num_actions)

    def select_action(self, state):
        state_in = tf.expand_dims(state, axis=0)
        probabilities = self.model(state_in)
        action = (tf.math.argmax(probabilities, 1)[0]).numpy()
        return action

    def get_model(self):
        return self.model


environment = make('hungry_geese')
a = Agent(rows=11, columns=11, num_actions=3)

# TODO: Enter / Change encoded weights string
encoded_weights = b''
decoded_weights = base64.b64decode(encoded_weights)
with io.BytesIO(decoded_weights) as f:
    weights = list(np.load(f, allow_pickle=True))
    f.close()
a.get_model().call(np.zeros((1, a.rows, a.columns, 1)))
a.get_model().set_weights(weights)

prev_direction = 0


def agent(obs_dict, config_dict):
    global environment, a, prev_direction

    state = preprocess_state(obs_dict, prev_direction)
    action = a.select_action(state)
    direction = get_direction(prev_direction, action)

    prev_direction = direction

    return environment.specification.action.enum[direction]
