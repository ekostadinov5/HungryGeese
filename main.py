
import os
import tensorflow as tf
from dqn import dqn_train
from ddqn import ddqn_train
from ppo import ppo_train
from selfplay import dqn_selfplay, ddqn_selfplay, ppo_selfplay


# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Limiting GPU memory growth
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


if __name__ == '__main__':
    # model_name = 'model'

    # dqn_train(model_name)
    # ddqn_train(model_name)
    # ppo_train(model_name)
    # dqn_selfplay(model_name)
    # ddqn_selfplay(model_name)
    # ppo_selfplay(model_name)

    pass
