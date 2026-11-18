import numpy as np

ECHELON_NAMES = ["Ana Depo", "Fabrika", "Tedarikçi"]


class DemandGenerator:
    def __init__(self, base=10.0, season_amp=2.5, season_period=26,
                 shock_prob=0.7, shock_size=(0.4, 0.8), trend_std=0.03):
        self.base = base
        self.season_amp = season_amp
        self.season_period = season_period
        self.shock_prob = shock_prob
        self.shock_size = shock_size
        self.trend_std = trend_std

    def sample(self, T, rng):
        t = np.arange(T)
        phase = rng.uniform(0, 2 * np.pi)
        mean = self.base + self.season_amp * np.sin(2 * np.pi * t / self.season_period + phase)
        mean = mean * (1 + rng.normal(0, self.trend_std) * t / 10)
        if rng.random() < self.shock_prob:
            start = rng.integers(8, T - 15)
            mean[start:] *= 1 + rng.uniform(*self.shock_size)
        mean = np.clip(mean, 1, None)
        return rng.poisson(mean).astype(float)


class LeadTimeProcess:
    def __init__(self, normal=(1, 2, 3), normal_p=(0.3, 0.5, 0.2),
                 crisis=(3, 4, 5), crisis_p=(0.3, 0.4, 0.3),
                 p_enter=0.04, p_exit=0.25):
        self.normal, self.normal_p = np.array(normal), np.array(normal_p)
        self.crisis, self.crisis_p = np.array(crisis), np.array(crisis_p)
        self.p_enter, self.p_exit = p_enter, p_exit

    def reset(self, n_links, rng):
        self.in_crisis = np.zeros(n_links, dtype=bool)
        self.rng = rng

    def step(self):
        r = self.rng.random(len(self.in_crisis))
        self.in_crisis = np.where(self.in_crisis, r > self.p_exit, r < self.p_enter)

    def sample(self, link):
        if self.in_crisis[link]:
            return int(self.rng.choice(self.crisis, p=self.crisis_p))
        return int(self.rng.choice(self.normal, p=self.normal_p))


class SupplyChainEnv:
    def __init__(self, T=52, n_echelons=3,
                 price=(4.0, 3.0, 2.0),
                 holding=(0.5, 0.4, 0.3),
                 backorder=(2.0, 1.6, 1.2),
                 smooth_penalty=0.25,
                 alpha_forecast=0.3,
                 demand=None, lead=None, init_inv=15, init_flow=10):
        self.T, self.n = T, n_echelons
        self.price, self.h, self.b = map(np.array, (price, holding, backorder))
        self.lam = smooth_penalty
        self.a_f = alpha_forecast
        self.demand_gen = demand or DemandGenerator()
        self.lead = lead or LeadTimeProcess()
        self.init_inv, self.init_flow = init_inv, init_flow

    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.t = 0
        self.demand = self.demand_gen.sample(self.T, rng)
        self.lead.reset(self.n, rng)
        n, f0 = self.n, self.init_flow
        self.inv = np.full(n, float(self.init_inv))
        self.backlog = np.zeros(n)
        self.pipeline = [np.zeros(8) for _ in range(n)]
        for i in range(n):
            self.pipeline[i][0:2] = f0
        self.on_order = np.full(n, 2.0 * f0)
        self.last_order = np.full(n, float(f0))
        self.forecast = np.full(n, float(f0))
        self.last_demand_in = np.full(n, float(f0))
        self.lt_est = np.full(n, 2.0)
        self.hist = {k: [] for k in ("orders", "inv", "backlog", "demand_in", "shipped")}
        return self._obs()

    def _obs(self):
        obs = []
        for i in range(self.n):
            ip = self.inv[i] - self.backlog[i] + self.on_order[i]
            obs.append(dict(
                inv=self.inv[i], backlog=self.backlog[i], on_order=self.on_order[i],
                inv_position=ip, forecast=self.forecast[i],
                last_demand=self.last_demand_in[i], lt_est=self.lt_est[i],
                last_order=self.last_order[i],
            ))
        return obs

    def step(self, orders):
        orders = np.maximum(0.0, np.round(np.asarray(orders, dtype=float)))
        n = self.n
        self.lead.step()

        for i in range(n):
            arrived = self.pipeline[i][0]
            self.pipeline[i] = np.roll(self.pipeline[i], -1)
            self.pipeline[i][-1] = 0.0
            self.inv[i] += arrived
            self.on_order[i] -= arrived

        demand_in = np.empty(n)
        demand_in[0] = self.demand[self.t]
        demand_in[1:] = self.last_order[:-1]

        shipped = np.zeros(n)
        for i in range(n):
            need = self.backlog[i] + demand_in[i]
            shipped[i] = min(self.inv[i], need)
            self.inv[i] -= shipped[i]
            self.backlog[i] = need - shipped[i]
            if i > 0 and shipped[i] > 0:
                L = self.lead.sample(i - 1)
                self.pipeline[i - 1][L - 1] += shipped[i]
                self.lt_est[i - 1] = 0.8 * self.lt_est[i - 1] + 0.2 * L

        top = n - 1
        if orders[top] > 0:
            L = self.lead.sample(top)
            self.pipeline[top][L - 1] += orders[top]
            self.lt_est[top] = 0.8 * self.lt_est[top] + 0.2 * L
        self.on_order += orders

        revenue = self.price * shipped
        hold = self.h * self.inv
        back = self.b * self.backlog
        smooth = self.lam * np.abs(orders - self.last_order)
        local_reward = revenue - hold - back - smooth

        self.forecast = self.a_f * demand_in + (1 - self.a_f) * self.forecast
        self.last_demand_in = demand_in
        self.last_order = orders
        for k, v in (("orders", orders), ("inv", self.inv.copy()), ("backlog", self.backlog.copy()),
                     ("demand_in", demand_in), ("shipped", shipped)):
            self.hist[k].append(np.array(v))

        self.t += 1
        done = self.t >= self.T
        info = dict(revenue=revenue, hold=hold, back=back, smooth=smooth,
                    customer_demand=demand_in[0], customer_served=shipped[0])
        return self._obs(), local_reward, done, info
