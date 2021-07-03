
import os
import io
import base64
import numpy as np
from kaggle_environments import make
from agents import QAgent, PPOAgent


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == '__main__':
    env = make('hungry_geese')
    agent = QAgent(rows=11, columns=11, num_actions=3)
    # agent = PPOAgent(rows=11, columns=11, num_actions=3)
    # TODO: Enter / Change model name
    full_model_name = ''
    agent.load_model_weights('models/' + full_model_name + '.h5')

    weights = agent.get_model().get_weights()
    # print(weights)
    np.save('model_weights.npy', weights)

    with open('model_weights.npy', 'rb') as f:
        b = f.read()
        # print(b)
        encoded_weights = base64.b64encode(b)
        # print(encoded_weights)
        with open('encoded_weights.txt', 'x') as w:
            w.write(str(encoded_weights))
            w.close()
        f.close()

    decoded_weights = base64.b64decode(encoded_weights)
    # print(decoded_weights)

    with io.BytesIO(decoded_weights) as f:
        result = list(np.load(f, allow_pickle=True))
        # print(result)
        f.close()

    agent.get_model().set_weights(result)
