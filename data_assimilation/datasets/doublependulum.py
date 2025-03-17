import numpy as np
import torch
from numpy import cos, sin
from numpy.random import MT19937, RandomState, SeedSequence
from scipy.integrate import solve_ivp
from torch.utils.data import Dataset

device = "cuda"

G = 9.8  # m/s^2
K1 = 0.5 * 0.4 * 1.26 * 4.2e-3  # baseball
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
L = L1 + L2  # maximal length of the combined pendulum
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg


def derivs(t, state):
    dydx = np.zeros_like(state)

    dydx[0] = state[1]

    delta = state[2] - state[0]
    den1 = (M1 + M2) * L1 - M2 * L1 * cos(delta) * cos(delta)
    dydx[1] = (
        M2 * L1 * state[1] * state[1] * sin(delta) * cos(delta)
        + M2 * G * sin(state[2]) * cos(delta)
        + M2 * L2 * state[3] * state[3] * sin(delta)
        - (M1 + M2) * G * sin(state[0])
    ) / den1

    dydx[2] = state[3]

    den2 = (L2 / L1) * den1
    dydx[3] = (
        -M2 * L2 * state[3] * state[3] * sin(delta) * cos(delta)
        + (M1 + M2) * G * sin(state[0]) * cos(delta)
        - (M1 + M2) * L1 * state[1] * state[1] * sin(delta)
        - (M1 + M2) * G * sin(state[2])
    ) / den2

    return dydx


class DoublePendulum(Dataset):
    def __init__(
        self,
        num_data,
        dt,
        n_steps,
        obs_noise,
        take_loss_physical=False,
        seed: int = 0,
        device="cuda",
    ):
        super().__init__()
        self.num_data = num_data
        self.dt = dt
        self.n_steps = n_steps
        self.obs_noise = obs_noise
        print(f"{seed=}")
        self.seed = seed
        self.rs = RandomState(MT19937(SeedSequence(seed)))
        self.take_loss_physical = take_loss_physical
        self.device = device

    @property
    def duration(self):
        return self.dt * self.n_steps

    def _generate_data(self):
        y_obs = []
        t = np.arange(0, self.dt * self.n_steps, self.dt)
        # th1 and th2 are the initial angles (degrees)
        # w10 and w20 are the initial angular velocities (degrees per second)
        (th1, th2) = self.rs.rand(2) * 360
        (w1, w2) = self.rs.rand(2) * 120 - 60
        # initial state
        state = np.radians([th1, w1, th2, w2])
        y = solve_ivp(derivs, t[[0, -1]], state, t_eval=t).y.T
        #y = ((y+np.pi) % (2*np.pi)) - np.pi # to limit between [-pi, pi]
        x1 = L1 * sin(y[:, 0])
        y1 = -L1 * cos(y[:, 0])
        x2 = L2 * sin(y[:, 2]) + x1
        y2 = -L2 * cos(y[:, 2]) + y1
        y_true = np.array([x1, y1, x2, y2])
        y_obs = y_true + self.obs_noise * self.rs.rand(4, self.n_steps)
        self.y_obs = y_obs.transpose(1, 0)
        assert y_true.shape == y_obs.shape
        if self.take_loss_physical:
            return [torch.from_numpy(self.y_obs).to(torch.float64).to(device),
                    torch.from_numpy(y).to(torch.float64).to(device)]  # m
        else:
            return [torch.from_numpy(self.y_obs).to(torch.float64).to(device)]  # m

    def __getitem__(self, index: int):
        assert index <= self.num_data, f"{index=}, {self.num_data=}"
        return self._generate_data()

    def __len__(self) -> int:
        return self.num_data
