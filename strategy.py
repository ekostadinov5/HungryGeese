
class EpsilonGreedyStrategy:

    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_epsilon(self, current_episode):
        epsilon = self.start - self.decay * current_episode
        return epsilon if epsilon > self.end else self.end
