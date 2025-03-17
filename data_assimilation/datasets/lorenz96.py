import numpy as np
import torch
from numpy.random import MT19937, RandomState, SeedSequence
from scipy.integrate import solve_ivp
from torch.utils.data import Dataset

G = 9.8  # m/s^2
K1 = 0.5 * 0.4 * 1.26 * 4.2e-3  # baseball


def lorenz96(t, X, forcing):
    N = len(X)
    dXdt = np.zeros(N)
    for i in range(N):
        dXdt[i] = (X[(i + 1) % N] - X[(i - 2) % N]) * X[(i - 1) % N] - X[i] + forcing

    return dXdt


class Lorenz96TimeSeries(Dataset):
    def __init__(
        self,
        num_data,
        dt,
        n_steps,
        time_series_length,
        obs_noise=0.0,
        n_grids=40,
        forcing=8,
        obs_data_complete="complete",
        thinout=False,
        take_loss_physical=False,
        seed: int = 0,
        device="cuda",
    ):
        super().__init__()
        self.num_data = num_data
        self.n_grids = n_grids
        self.dt = dt
        self.n_steps = n_steps
        self.time_series_length = time_series_length
        self.obs_noise = obs_noise
        self.forcing = forcing
        self.obs_data_complete = obs_data_complete
        self.thinout = thinout
        self.data = [None] * num_data
        self.take_loss_physical = take_loss_physical
        self.seed = seed
        self.rs = RandomState(MT19937(SeedSequence(seed)))
        self.device = device
        self.spinup = 20

    @property
    def duration(self):
        return self.dt * self.n_steps

    def _generate_data(self):
        z0 = self.rs.rand(self.n_grids) * 20 - 10
        ts = np.array(
            [self.dt * i for i in range(self.n_steps + self.time_series_length + self.spinup)]
        )
        y_true = solve_ivp(
            fun=lorenz96,
            t_span=[0, self.dt * (self.n_steps + self.time_series_length + self.spinup)],
            y0=z0,
            t_eval=ts,
            rtol=1e-2,
            atol=1e-2,
            args=(self.forcing,),
        ).y
        assert y_true.shape == (self.n_grids, self.n_steps + self.time_series_length + self.spinup)

        error = self.rs.multivariate_normal(
            np.zeros(self.n_grids),
            self.obs_noise * self.obs_noise * np.eye(self.n_grids),
            size=self.n_steps + self.time_series_length + self.spinup,
        )

        if self.obs_data_complete == "quad_capped_1000":
            y_obs = y_true * y_true * y_true * y_true + error.T
            y_obs = np.where(y_obs < 1000, y_obs, 1000) / 100
        elif self.obs_data_complete == "quad_capped_100":
            y_obs = y_true * y_true * y_true * y_true + error.T
            y_obs = np.where(y_obs < 100, y_obs, 100) / 100
        elif self.obs_data_complete == "quad_capped_10":
            y_obs = y_true * y_true * y_true * y_true
            y_obs = (np.where(y_obs < 10, y_obs, 10) + error.T) / 10
        elif self.obs_data_complete == "powerto6_capped_1000":
            y_obs = y_true * y_true * y_true * y_true * y_true * y_true + error.T
            y_obs = np.where(y_obs < 1000, y_obs, 1000) / 100
        elif self.obs_data_complete == "complete":
            y_obs = y_true + error.T
        else:
            y_obs = y_true + error.T

        y_true_folded = y_true[
            :,
            np.arange(self.time_series_length)[:, np.newaxis]
            + np.arange(self.spinup + self.n_steps + 1),
        ]
        y_obs_folded = y_obs[
            :,
            np.arange(self.time_series_length)[:, np.newaxis]
            + np.arange(self.spinup + self.n_steps + 1),
        ]
        y_true_folded = np.transpose(y_true_folded, (1, 0, 2))
        y_obs_folded = np.transpose(y_obs_folded, (1, 0, 2))

        assert y_true_folded.shape == (
            self.time_series_length,
            self.n_grids,
            self.n_steps + self.spinup + 1,
        )
        assert y_obs_folded.shape == (
            self.time_series_length,
            self.n_grids,
            self.n_steps + self.spinup + 1,
        )

        if self.thinout:
            y_obs_folded = y_obs_folded[::2, :, :]

        if self.obs_data_complete == "incomplete_half":
            y_obs_folded = y_obs_folded[:, : self.n_grids // 2, :]
        elif self.obs_data_complete == "incomplete_sparse":
            y_obs_folded = y_obs_folded[:, ::4, :]

        self.y_true = y_true
        y_obs_folded = y_obs_folded[
            :, :, self.spinup : self.n_steps + self.spinup
        ]  # cut one step for shape consistency
        y_true_folded = y_true_folded[
            :, :, self.spinup : self.n_steps + self.spinup
        ]  # cut one step for shape consistency
        self.y_obs_folded = y_obs_folded

        if self.take_loss_physical:
            return [
                torch.from_numpy(y_obs_folded)
                .transpose(0, 2)
                .transpose(1, 2)
                .to(torch.float64)
                .to(self.device),
                torch.from_numpy(y_true_folded)
                .transpose(0, 2)
                .transpose(1, 2)
                .to(torch.float64)
                .to(self.device),
            ]
        else:
            return [
                torch.from_numpy(y_obs_folded)
                .transpose(0, 2)
                .transpose(1, 2)
                .to(torch.float64)
                .to(self.device)
            ]  # should be n_steps, n_grids, time_series_length

    def __getitem__(self, index: int):
        assert index <= self.num_data, f"{index=}, {self.num_data=}"
        return self._generate_data()

    def __len__(self) -> int:
        return self.num_data
