
import numpy as np
from matplotlib import pyplot as plt
from kaggle_environments import make
from agents import DQNAgent
from memory import ReplayBuffer
from strategy import EpsilonGreedyStrategy
from utils import preprocess_state, calculate_reward, get_direction


def dqn_train(model_name, load_model=False, model_filename=None, optimizer_filename=None):
    print("DQN -- Training")

    env = make('hungry_geese')
    trainer = env.train(['greedy', None, 'agents/boilergoose.py', 'agents/handy_rl.py'])

    agent = DQNAgent(rows=11, columns=11, num_actions=3)
    buffer = ReplayBuffer()
    strategy = EpsilonGreedyStrategy(start=0.5, end=0.0, decay=0.00001)

    if load_model:
        agent.load_model_weights(model_filename)
        agent.load_optimizer_weights(optimizer_filename)

    start_episode = 0
    end_episode = 50000
    epochs = 32
    batch_size = 128

    training_rewards = []
    evaluation_rewards = []
    last_1000_ep_reward = []

    for episode in range(start_episode + 1, end_episode + 1):
        obs_dict = trainer.reset()
        epsilon = strategy.get_epsilon(episode - start_episode)
        ep_reward, ep_steps, done = 0, 0, False
        prev_direction = 0

        while not done:
            ep_steps += 1

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

        print("EPISODE " + str(episode) + " - REWARD: " + str(ep_reward) + " - STEPS: " + str(ep_steps))

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

        if episode % 1000 == 0:
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

        if episode % 5000 == 0:
            agent.save_model_weights('models/dqn_' + model_name + '_' + str(episode) + '.h5')
            agent.save_optimizer_weights('models/dqn_' + model_name + '_' + str(episode) + '_optimizer.npy')

    agent.save_model_weights('models/dqn_' + model_name + '_' + str(end_episode) + '.h5')
    agent.save_optimizer_weights('models/dqn_' + model_name + '_' + str(end_episode) + '_optimizer.npy')

    plt.plot([i for i in range(start_episode + 1000, end_episode + 1, 1000)], training_rewards)
    plt.title('Reward')
    plt.show()

    plt.plot([i for i in range(start_episode + 1000, end_episode + 1, 1000)], evaluation_rewards)
    plt.title('Evaluation rewards')
    plt.show()
