
# import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from kaggle_environments import make, evaluate
from agents import DQNAgent, DDQNAgent
from memory import ReplayBuffer
from strategy import EpsilonGreedyStrategy
from utils import preprocess_state, calculate_reward, get_direction


# Disable GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

# Limiting GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def train(agent, trained_model_name):
    env = make('hungry_geese')
    trainer = env.train(['greedy', None, 'agents/agent_dqn_100000.py', 'agents/agent_ddqn_100000.py'])
    agent = agent
    buffer = ReplayBuffer()

    # model_name = ''
    # agent.load_model_weights('models/' + model_name + '.h5')
    # agent.load_optimizer_weights('models/' + model_name + '_optimizer.npy')

    start_episode = 0
    end_episode = 100000
    epochs = 32
    batch_size = 128
    strategy = EpsilonGreedyStrategy(0.5, 0.0, 0.000005)

    training_rewards = []
    evaluation_rewards = []
    last_1000_ep_reward = []

    for episode in range(start_episode + 1, end_episode + 1):
        obs_dict = trainer.reset()
        epsilon = strategy.get_epsilon(episode - start_episode)
        ep_reward, done = 0, False
        prev_direction = 0

        while not done:
            state = preprocess_state(obs_dict, prev_direction)
            action = agent.select_epsilon_greedy_action(state, epsilon)
            direction = get_direction(prev_direction, action)
            next_obs_dict, _, done, _ = trainer.step(env.specification.action.enum[direction])
            reward = calculate_reward(obs_dict, next_obs_dict)
            next_state = preprocess_state(next_obs_dict, direction)
            buffer.add(state, action, reward, next_state, done)

            obs_dict = next_obs_dict
            prev_direction = direction

            ep_reward += reward

        if len(buffer) >= batch_size:
            for _ in range(epochs):
                states, actions, rewards, next_states, dones = buffer.get_samples(batch_size)
                agent.fit(states, actions, rewards, next_states, dones)

        if len(last_1000_ep_reward) == 1000:
            last_1000_ep_reward = last_1000_ep_reward[1:]
        last_1000_ep_reward.append(ep_reward)

        if episode % 10 == 0:
            agent.update_target_network()

        if episode % 1000 == 0:
            print('Episode ' + str(episode) + '/' + str(end_episode))
            print('Epsilon: ' + str(round(epsilon, 3)))

            last_1000_ep_reward_mean = np.mean(last_1000_ep_reward).round(3)
            training_rewards.append(last_1000_ep_reward_mean)
            print('Average reward in last 1000 episodes: ' + str(last_1000_ep_reward_mean))

            print()

        if episode % 10000 == 0:
            eval_reward = 0
            for i in range(100):
                obs_dict = trainer.reset()
                epsilon = 0
                done = False
                prev_direction = 0
                while not done:
                    state = preprocess_state(obs_dict, prev_direction)
                    action = agent.select_epsilon_greedy_action(state, epsilon)
                    direction = get_direction(prev_direction, action)
                    next_obs_dict, _, done, _ = trainer.step(env.specification.action.enum[direction])
                    reward = calculate_reward(obs_dict, next_obs_dict)
                    obs_dict = next_obs_dict
                    prev_direction = direction
                    eval_reward += reward
            eval_reward /= 100
            evaluation_rewards.append(eval_reward)
            print("Evaluation reward: " + str(eval_reward))

            print()

    trained_model_name = trained_model_name
    agent.save_model_weights('models/' + trained_model_name + '_' + str(end_episode) + '.h5')
    agent.save_optimizer_weights('models/' + trained_model_name + '_' + str(end_episode) + '_optimizer.npy')

    plt.plot([i for i in range(start_episode + 1000, end_episode + 1, 1000)], training_rewards)
    plt.title('Reward')
    plt.show()

    plt.plot([i for i in range(start_episode + 10000, end_episode + 1, 10000)], evaluation_rewards)
    plt.title('Evaluation rewards')
    plt.show()


def dqn_train():
    print("DQN -- Training")
    agent = DQNAgent(rows=11, columns=11, num_actions=3)
    train(agent, 'dqn')


def ddqn_train():
    print("DDQN -- Training")
    agent = DDQNAgent(rows=11, columns=11, num_actions=3)
    train(agent, 'ddqn')


def eval():
    scores = evaluate('hungry_geese', ['greedy', 'submission.py', 'greedy', 'greedy'], num_episodes=100)
    scoreboard = [0, 0, 0, 0]
    for score in scores:
        winner = np.argmax(score)
        scoreboard[winner] += 1
    print(scores)
    print(scoreboard)
    print()


if __name__ == '__main__':
    # dqn_train()
    # ddqn_train()
    # eval()
    pass
