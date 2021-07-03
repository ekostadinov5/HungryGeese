
import numpy as np
from matplotlib import pyplot as plt
from kaggle_environments import make
from agents import PPOAgent
from memory import Memory
from utils import preprocess_state, calculate_reward, get_direction


def ppo_train(model_name, load_model=False, actor_filename=None, critic_filename=None, optimizer_filename=None):
    print("PPO -- Training")

    env = make('hungry_geese')
    trainer = env.train(['greedy', None, 'agents/boilergoose.py', 'agents/handy_rl.py'])

    agent = PPOAgent(rows=11, columns=11, num_actions=3)
    memory = Memory()

    if load_model:
        agent.load_model_weights(actor_filename, critic_filename)
        agent.load_optimizer_weights(optimizer_filename)

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
        obs_dict = trainer.reset()
        ep_reward, ep_steps, done = 0, 0, False
        prev_direction = 0

        while not done:
            current_frame += 1
            ep_steps += 1

            state = preprocess_state(obs_dict, prev_direction)
            action = agent.select_action(state, training=True)
            direction = get_direction(prev_direction, action)
            next_obs_dict, _, done, _ = trainer.step(env.specification.action.enum[direction])
            reward = calculate_reward(obs_dict, next_obs_dict)
            next_state = preprocess_state(next_obs_dict, direction)
            memory.add(state, action, reward, next_state, float(done))

            obs_dict = next_obs_dict
            prev_direction = direction

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
                    agent.save_model_weights('models/ppo_actor_' + model_name + '_' + str(episode) + '.h5',
                                             'models/ppo_critic_' + model_name + '_' + str(episode) + '.h5')
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
            agent.save_model_weights('models/ppo_actor_' + model_name + '_' + str(episode) + '.h5',
                                     'models/ppo_critic_' + model_name + '_' + str(episode) + '.h5')
            agent.save_optimizer_weights('models/ppo_' + model_name + '_' + str(episode) + '_optimizer.npy')

    agent.save_model_weights('models/ppo_actor_' + model_name + '_' + str(end_episode) + '.h5',
                             'models/ppo_critic_' + model_name + '_' + str(end_episode) + '.h5')
    agent.save_optimizer_weights('models/ppo_' + model_name + '_' + str(end_episode) + '_optimizer.npy')

    if threshold_reached:
        plt.plot([i for i in range(start_episode + 1000, episode, 1000)], training_rewards)
    else:
        plt.plot([i for i in range(start_episode + 1000, end_episode + 1, 1000)], training_rewards)
    plt.title("Reward")
    plt.show()

    plt.plot([i for i in range(start_episode + 1000, end_episode + 1, 1000)], evaluation_rewards)
    plt.title('Evaluation rewards')
    plt.show()
