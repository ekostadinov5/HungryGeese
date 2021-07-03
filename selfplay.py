
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
from kaggle_environments import make
from agents import DQNAgent, DDQNAgent, PPOAgent
from memory import ReplayBuffer, Memory
from strategy import EpsilonGreedyStrategy
from utils import preprocess_state, calculate_reward, get_direction


def dqn_selfplay(model_name, load_model=False, model_filename=None, optimizer_filename=None):
    print("DQN -- Self-play training")

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

    enemies = [deepcopy(agent), deepcopy(agent), deepcopy(agent)]

    for episode in range(start_episode + 1, end_episode + 1):
        obs_dict = env.reset(4)
        obs_dict = obs_dict[0].observation
        epsilon = strategy.get_epsilon(episode - start_episode)
        ep_reward, ep_steps, done = 0, 0, False
        prev_direction = 0
        enemies_prev_direction = [0, 0, 0]

        while not done:
            ep_steps += 1

            state = preprocess_state(obs_dict, prev_direction)
            action = agent.select_epsilon_greedy_action(state, epsilon)
            direction = get_direction(prev_direction, action)

            enemies_obs_dict = deepcopy(obs_dict)
            enemies_direction = []
            for index, enemy, enemy_prev_direction in zip(range(3), enemies, enemies_prev_direction):
                enemies_obs_dict['index'] = index + 1
                enemy_state = preprocess_state(enemies_obs_dict, enemy_prev_direction)
                enemy_action = enemy.select_action(enemy_state)
                enemy_direction = get_direction(enemy_prev_direction, enemy_action)
                enemies_direction.append(enemy_direction)

            step = env.step([
                env.specification.action.enum[direction],
                env.specification.action.enum[enemies_direction[0]],
                env.specification.action.enum[enemies_direction[1]],
                env.specification.action.enum[enemies_direction[2]]
            ])
            next_obs_dict, _, done = step[0].observation, (step[0].reward - ep_reward), step[0].status == 'DONE'
            reward = calculate_reward(obs_dict, next_obs_dict)
            next_state = preprocess_state(next_obs_dict, direction)
            buffer.add(state, action, reward, next_state, done)

            obs_dict = next_obs_dict
            prev_direction = direction
            enemies_prev_direction = enemies_direction

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
            agent.save_model_weights('models/self-play_dqn_' + model_name + '_' + str(episode) + '.h5')
            agent.save_optimizer_weights('models/self-play_dqn_' + model_name + '_' + str(episode) + '_optimizer.npy')

        if episode % 5000 == 0:
            enemies = enemies[1:]
            enemies.append(deepcopy(agent))

    agent.save_model_weights('models/self-play_dqn_' + model_name + '_' + str(end_episode) + '.h5')
    agent.save_optimizer_weights('models/self-play_dqn_' + model_name + '_' + str(end_episode) + '_optimizer.npy')

    plt.plot([i for i in range(start_episode + 1000, end_episode + 1, 1000)], training_rewards)
    plt.title('Reward')
    plt.show()

    plt.plot([i for i in range(start_episode + 1000, end_episode + 1, 1000)], evaluation_rewards)
    plt.title('Evaluation rewards')
    plt.show()


def ddqn_selfplay(model_name, load_model=False, model_filename=None, optimizer_filename=None):
    print("DDQN -- Self-play training")

    env = make('hungry_geese')
    trainer = env.train(['greedy', None, 'agents/boilergoose.py', 'agents/handy_rl.py'])

    agent = DDQNAgent(rows=11, columns=11, num_actions=3)
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

    enemies = [deepcopy(agent), deepcopy(agent), deepcopy(agent)]

    for episode in range(start_episode + 1, end_episode + 1):
        obs_dict = env.reset(4)
        obs_dict = obs_dict[0].observation
        epsilon = strategy.get_epsilon(episode - start_episode)
        ep_reward, ep_steps, done = 0, 0, False
        prev_direction = 0
        enemies_prev_direction = [0, 0, 0]

        while not done:
            ep_steps += 1

            state = preprocess_state(obs_dict, prev_direction)
            action = agent.select_epsilon_greedy_action(state, epsilon)
            direction = get_direction(prev_direction, action)

            enemies_obs_dict = deepcopy(obs_dict)
            enemies_direction = []
            for index, enemy, enemy_prev_direction in zip(range(3), enemies, enemies_prev_direction):
                enemies_obs_dict['index'] = index + 1
                enemy_state = preprocess_state(enemies_obs_dict, enemy_prev_direction)
                enemy_action = enemy.select_action(enemy_state)
                enemy_direction = get_direction(enemy_prev_direction, enemy_action)
                enemies_direction.append(enemy_direction)

            step = env.step([
                env.specification.action.enum[direction],
                env.specification.action.enum[enemies_direction[0]],
                env.specification.action.enum[enemies_direction[1]],
                env.specification.action.enum[enemies_direction[2]]
            ])
            next_obs_dict, _, done = step[0].observation, (step[0].reward - ep_reward), step[0].status == 'DONE'
            reward = calculate_reward(obs_dict, next_obs_dict)
            next_state = preprocess_state(next_obs_dict, direction)
            buffer.add(state, action, reward, next_state, done)

            obs_dict = next_obs_dict
            prev_direction = direction
            enemies_prev_direction = enemies_direction

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
            agent.save_model_weights('models/self-play_ddqn_' + model_name + '_' + str(episode) + '.h5')
            agent.save_optimizer_weights('models/self-play_ddqn_' + model_name + '_' + str(episode) + '_optimizer.npy')

        if episode % 5000 == 0:
            enemies = enemies[1:]
            enemies.append(deepcopy(agent))

    agent.save_model_weights('models/self-play_ddqn_' + model_name + '_' + str(end_episode) + '.h5')
    agent.save_optimizer_weights('models/self-play_ddqn_' + model_name + '_' + str(end_episode) + '_optimizer.npy')

    plt.plot([i for i in range(start_episode + 1000, end_episode + 1, 1000)], training_rewards)
    plt.title('Reward')
    plt.show()

    plt.plot([i for i in range(start_episode + 1000, end_episode + 1, 1000)], evaluation_rewards)
    plt.title('Evaluation rewards')
    plt.show()


def ppo_selfplay(model_name, load_model=False, actor_filename=None, critic_filename=None, optimizer_filename=None):
    print("PPO -- Self-play training")

    env = make('hungry_geese')
    trainer = env.train(['greedy', None, 'agents/boilergoose.py', 'agents/handy_rl.py'])

    agent = PPOAgent(rows=11, columns=11, num_actions=3)
    memory = Memory()

    if load_model:
        agent.load_model_weights(actor_filename, critic_filename)
        agent.load_optimizer_weights(optimizer_filename)

    enemies = [deepcopy(agent), deepcopy(agent), deepcopy(agent)]

    episode = 0
    start_episode = 0
    end_episode = 50000
    reward_threshold = None
    threshold_reached = False
    epochs = 4
    batch_size = 128
    current_frame = 0

    training_rewards = []
    evaluation_rewards = []
    last_1000_ep_reward = []

    for episode in range(start_episode + 1, end_episode + 1):
        obs_dict = env.reset(4)
        obs_dict = obs_dict[0].observation
        ep_reward, ep_steps, done = 0, 0, False
        prev_direction = 0
        enemies_prev_direction = [0, 0, 0]

        while not done:
            current_frame += 1
            ep_steps += 1

            state = preprocess_state(obs_dict, prev_direction)
            action = agent.select_action(state, training=True)
            direction = get_direction(prev_direction, action)

            enemies_obs_dict = deepcopy(obs_dict)
            enemies_direction = []
            for index, enemy, enemy_prev_direction in zip(range(3), enemies, enemies_prev_direction):
                enemies_obs_dict['index'] = index + 1
                enemy_state = preprocess_state(enemies_obs_dict, enemy_prev_direction)
                enemy_action = enemy.select_action(enemy_state)
                enemy_direction = get_direction(enemy_prev_direction, enemy_action)
                enemies_direction.append(enemy_direction)

            step = env.step([
                env.specification.action.enum[direction],
                env.specification.action.enum[enemies_direction[0]],
                env.specification.action.enum[enemies_direction[1]],
                env.specification.action.enum[enemies_direction[2]]
            ])
            next_obs_dict, _, done = step[0].observation, (step[0].reward - ep_reward), step[0].status == 'DONE'
            reward = calculate_reward(obs_dict, next_obs_dict)
            next_state = preprocess_state(next_obs_dict, direction)
            memory.add(state, action, reward, next_state, float(done))

            obs_dict = next_obs_dict
            prev_direction = direction
            enemies_prev_direction = enemies_direction

            ep_reward += reward

            if current_frame % batch_size == 0:
                for _ in range(epochs):
                    states, actions, rewards, next_states, dones = memory.get_all_samples()
                    agent.fit(states, actions, rewards, next_states, dones)
                memory.clear()
                agent.update_networks()

        print("EPISODE " + str(episode) + " - REWARD: " + str(ep_reward) + " - STEPS: " + str(ep_steps))

        if len(last_1000_ep_reward) == 1000:
            last_1000_ep_reward = last_1000_ep_reward[1:]
        last_1000_ep_reward.append(ep_reward)

        if reward_threshold:
            if len(last_1000_ep_reward) == 1000:
                if np.mean(last_1000_ep_reward) >= reward_threshold:
                    print("You solved the task after" + str(episode) + "episodes")
                    agent.save_model_weights('models/self-play_ppo_actor_' + model_name + '_' + str(episode) + '.h5',
                                             'models/self-play_ppo_critic_' + model_name + '_' + str(episode) + '.h5')
                    threshold_reached = True
                    break

        if episode % 1000 == 0:
            print('Episode ' + str(episode) + '/' + str(end_episode))

            last_1000_ep_reward_mean = np.mean(last_1000_ep_reward).round(3)
            training_rewards.append(last_1000_ep_reward_mean)
            print('Average reward in last 1000 episodes: ' + str(last_1000_ep_reward_mean))
            print()

        if episode % 1000 == 0:
            eval_reward = 0
            for i in range(100):
                obs_dict = trainer.reset()
                done = False
                prev_direction = 0
                while not done:
                    state = preprocess_state(obs_dict, prev_direction)
                    action = agent.select_action(state)
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
            agent.save_model_weights('models/self-play_ppo_actor_' + model_name + '_' + str(episode) + '.h5',
                                     'models/self-play_ppo_critic_' + model_name + '_' + str(episode) + '.h5')
            agent.save_optimizer_weights('models/self-play_ppo_' + model_name + '_' + str(episode) + '_optimizer.npy')

        if episode % 5000 == 0:
            enemies = enemies[1:]
            enemies.append(deepcopy(agent))

    agent.save_model_weights('models/self-play_ppo_actor_' + model_name + '_' + str(end_episode) + '.h5',
                             'models/self-play_ppo_critic_' + model_name + '_' + str(end_episode) + '.h5')
    agent.save_optimizer_weights('models/self-play_ppo_' + model_name + '_' + str(end_episode) + '_optimizer.npy')

    if threshold_reached:
        plt.plot([i for i in range(start_episode + 1000, episode, 1000)], training_rewards)
    else:
        plt.plot([i for i in range(start_episode + 1000, end_episode + 1, 1000)], training_rewards)
    plt.title("Reward")
    plt.show()

    plt.plot([i for i in range(start_episode + 1000, end_episode + 1, 1000)], evaluation_rewards)
    plt.title('Evaluation rewards')
    plt.show()
