
from kaggle_environments import make
from agents import Agent
from utils import preprocess_state, get_direction


prev_direction = 0


def agent(obs_dict, config_dict):
    global prev_direction

    env = make('hungry_geese')
    agent = Agent(rows=11, columns=11, num_actions=3)
    model_name = ''
    agent.load_model_weights('models/' + model_name + '.h5')

    state = preprocess_state(obs_dict, prev_direction)
    action = agent.select_action(state)
    direction = get_direction(prev_direction, action)
    prev_direction = direction
    return env.specification.action.enum[direction]
