import numpy as np

DELTAS = np.array([-8, -4, -2, 0, 2, 4, 8], dtype=float)

GAP_BINS = np.array([-25, -12, -5, 0, 5, 12, 25])
TREND_BINS = np.array([-0.15, 0.15])
LT_BINS = np.array([2.6])
BACKLOG_BINS = np.array([0.5])
N_STATES = (len(GAP_BINS) + 1) * (len(TREND_BINS) + 1) * (len(LT_BINS) + 1) * (len(BACKLOG_BINS) + 1)
N_ACTIONS = len(DELTAS)


def discretize(o):
    target = o["forecast"] * (o["lt_est"] + 1)
    gap = o["inv_position"] - target
    trend = (o["last_demand"] - o["forecast"]) / max(o["forecast"], 1.0)
    g = int(np.digitize(gap, GAP_BINS))
    tr = int(np.digitize(trend, TREND_BINS))
    lt = int(np.digitize(o["lt_est"], LT_BINS))
    bk = int(np.digitize(o["backlog"], BACKLOG_BINS))
    return ((g * 3 + tr) * 2 + lt) * 2 + bk


def action_to_order(o, a):
    return max(0.0, round(o["forecast"] + DELTAS[a]))


class TabularAgent:
    def __init__(self, method="qlearning", gamma=0.95, alpha=0.1, alpha_min=0.01,
                 eps=1.0, eps_min=0.02, seed=0):
        assert method in ("qlearning", "sarsa")
        self.method = method
        self.gamma = gamma
        self.alpha0, self.alpha_min = alpha, alpha_min
        self.eps0, self.eps_min = eps, eps_min
        self.Q = np.zeros((N_STATES, N_ACTIONS))
        self.visits = np.zeros((N_STATES, N_ACTIONS))
        self.rng = np.random.default_rng(seed)
        self.eps = eps

    def set_schedule(self, progress):
        self.eps = max(self.eps_min, self.eps0 * (1 - progress / 0.8))

    def act(self, s, greedy=False):
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.integers(N_ACTIONS))
        q = self.Q[s]
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def update(self, s, a, r, s2, a2, done):
        self.visits[s, a] += 1
        alpha = max(self.alpha_min, self.alpha0 / (1 + 0.002 * self.visits[s, a]))
        if done:
            target = r
        elif self.method == "qlearning":
            target = r + self.gamma * self.Q[s2].max()
        else:
            target = r + self.gamma * self.Q[s2, a2]
        self.Q[s, a] += alpha * (target - self.Q[s, a])


class RLPolicy:
    def __init__(self, agents, name):
        self.agents, self.name = agents, name

    def __call__(self, obs):
        return [action_to_order(o, ag.act(discretize(o), greedy=True))
                for o, ag in zip(obs, self.agents)]


def pass_through(obs):
    return [o["last_demand"] for o in obs]


def order_up_to(obs, safety=1.0):
    out = []
    for o in obs:
        S = o["forecast"] * (o["lt_est"] + 1) + safety * o["forecast"]
        out.append(max(0.0, round(S - o["inv_position"])))
    return out


def smoothed_order_up_to(obs, beta=0.3, safety=1.0):
    out = []
    for o in obs:
        S = o["forecast"] * (o["lt_est"] + 1) + safety * o["forecast"]
        out.append(max(0.0, round(o["forecast"] + beta * (S - o["inv_position"]))))
    return out
