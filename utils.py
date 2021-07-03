
import numpy as np


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


def calculate_reward(prev_obs_dict, obs_dict):
    index = obs_dict['index']
    goose = obs_dict['geese'][index]
    if not goose:
        return -70

    prev_goose = prev_obs_dict['geese'][index]
    if len(goose) > len(prev_goose):
        return 10

    return 1


def get_direction(prev_direction, action):
    new_direction = prev_direction + (0 if action == 0 else -1 if action == 1 else 1)
    if new_direction < 0:
        return 3
    if new_direction > 3:
        return 0
    return new_direction
