import os
from typing import List  # NOQA

import time
import numpy as np
import torch
import torch.distributions as dist
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
#import torchviz

from data_assimilation.models.doublependulum import network_init_doublependulum
from data_assimilation.models.lorenz96_assimilation_op import network_init_Lorenz96, network_init_Lorenz96_kernelsizes
from data_assimilation.models.moving_light_1D import network_init_moving_light_1D
from data_assimilation.models.moving_MNIST_2D import network_init_moving_MNIST_2D
from data_assimilation.models.moving_MNIST_2D_model_agnostic import (
    network_init_moving_MNIST_2D_model_agnostic,
)
from data_assimilation.models.moving_MNIST_3D import network_init_moving_MNIST_3D
from data_assimilation.utils import compute_K, compute_K_sigma_block_diag
from einops import rearrange

import time
from torch import profiler  # NOQA

torch.set_default_dtype(torch.float64)

def periodic_difference(x, y, period):
    d1 = (x-y)%period
    d2 = (y-x)%period
    return torch.where(abs(d1)<abs(d2), d1, d2)

def compute_loss_integral_VonMises(
    h_results,
    x_value,
    sigma_err,
    periodic_indices,
    nonperiodic_indices,
    concentration_periodic,
):
    #print(f"{concentration_periodic=}")
    (N_data, N_sample, n_step, x_dim) = h_results.shape
    h_results_periodic = h_results[:, :, :, periodic_indices]
    x_value_periodic = x_value[:, :, periodic_indices]
    h_results_nonperiodic = h_results[:, :, :, nonperiodic_indices]
    x_value_nonperiodic = x_value[:, :, nonperiodic_indices]

    #print(f"{h_results_periodic=}")
    #print(f"{h_results_nonperiodic=}")
    #print(f"{x_value_periodic=}")
    #print(f"{x_value_nonperiodic=}")
    
    vonmises_dist = dist.von_mises.VonMises(h_results_periodic, concentration_periodic, validate_args=None)
    quad_log_prob_periodic = torch.sum(vonmises_dist.log_prob(x_value_periodic.unsqueeze(1).repeat(1, N_sample, 1, 1)))
    #quad_log_prob_periodic = -0.5 * (periodic_difference(x=h_results_periodic, y=x_value_periodic.unsqueeze(1).repeat(1, N_sample, 1, 1), period=period) / sigma_err)**2
    gaussian_dist = dist.normal.Normal(h_results_nonperiodic, sigma_err)
    quad_log_prob_others = torch.sum(gaussian_dist.log_prob(x_value_nonperiodic.unsqueeze(1).repeat(1, N_sample, 1, 1)))#-0.5 * ((h_results_nonperiodic-x_value_nonperiodic.unsqueeze(1).repeat(1, N_sample, 1, 1)) / sigma_err)**2
    log_prob_values3 = torch.sum(quad_log_prob_periodic)+torch.sum(quad_log_prob_others)# - N_data*N_sample*n_step*len(nonperiodic_indices)*torch.log(sigma_err)
    #print(f"{torch.sum(quad_log_prob_periodic)=}, {torch.sum(quad_log_prob_others)=}, {log_prob_values3=}")    
    #print(x_value.shape)
    return log_prob_values3 / N_sample / N_data

def compute_loss_integral4(
    h_results,
    x_value,
    sigma_err,
):
    #print(x_value.shape)
    (N_data, N_sample, n_step, x_dim) = h_results.shape
    gaussian_dist = dist.normal.Normal(h_results, sigma_err)
    quad_log_prob = torch.sum(gaussian_dist.log_prob(x_value.unsqueeze(1).repeat(1, N_sample, 1, 1)))
    #quad_log_prob = -0.5 * ((h_results-x_value.unsqueeze(1).repeat(1, N_sample, 1, 1)) / sigma_err)**2
    log_prob_values3 = torch.sum(quad_log_prob)# - N_data*N_sample*n_step*x_dim*torch.log(sigma_err)
    return log_prob_values3 / N_sample / N_data

def compute_loss_integral3(
    mu_1,
    sigma_1,
    sigma_2,
    h,
    z_dim,
    x_dim,  # obs_dim or physical_dim depending on take_loss_physical
    x_value,
    N_data,
    n_step,
    sigma1_block_diag=False,
    periodic=False,
    periodic_indices=[],
    period=2*torch.pi,
    concentration_periodic=1e4,
):
    #torch.cuda.synchronize()
    #integral2_begin = time.time()
    
    if sigma1_block_diag:
        assert mu_1.shape == (
            N_data,
            n_step,
            z_dim//2,
            2
        ), f"mu_1.shape should be (N_data, n_step, z_dim//2, 2) but is {mu_1.shape}"
        assert sigma_1.shape == (
            N_data,
            n_step,
            z_dim//2,
            2,
            2,
        ), f"sigma_1.shape should be (N_data, n_step, z_dim//2, 2, 2) but is {sigma_1.shape}"
    else:
        assert mu_1.shape == (
            N_data,
            n_step,
            z_dim,
        ), f"mu_1.shape should be (N_data, n_step, z_dim) but is {mu_1.shape}"
        assert sigma_1.shape == (
            N_data,
            n_step,
            z_dim,
            z_dim,
        ), f"sigma_1.shape should be (N_data, n_step, z_dim, z_dim) but is {sigma_1.shape}"
        assert sigma_2.shape == (
            x_dim,
            x_dim,
        ), f"sigma_2.shape should be (x_dim, x_dim) but is {sigma_2.shape}"
        assert x_value.shape == (
            N_data,
            n_step,
            x_dim,
        ), f"x_value.shape should be (N_data, n_step, x_dim) but is {x_value.shape}"
    if periodic:
        nonperiodic_indices = list(set([i for i in range(x_dim)]) - set(periodic_indices))
        #print(f"{periodic_indices=}")
        #print(f"{nonperiodic_indices=}")

    assert ~torch.isnan(sigma_1).any(), f"sigma_1 contains nan: {sigma_1}"
    if sigma1_block_diag:
        sqrt_sigma_1 = torch.linalg.cholesky(sigma_1)
        mvn1 = dist.Normal(torch.tensor(0.0, device="cuda"), torch.tensor(1.0, device="cuda"))
        N_sample = 1
        points = mvn1.sample((N_data, n_step, N_sample, z_dim//2, 2))
        result = mu_1.unsqueeze(2).repeat(1, 1, N_sample, 1, 1) + torch.einsum(
            "abxij,absxj->absxi", sqrt_sigma_1, points
        )  # (N_data, n_step, N_sample, z_dim//2, 2)
        h_arg = result.transpose(1, 2) # (N_data, N_sample, n_step, z_dim//2, 2)
        h_arg = rearrange(h_arg, "d s st z1 z2 -> (d s st) (z1 z2)")
        #torch.cuda.synchronize()
        #reparametrization_end = time.time()
    
        h_results_ = h(h_arg)

        #torch.cuda.synchronize()
        #observation_end = time.time()
    
        log_prob_values3 = 0.0
        sigma_err = torch.exp(sigma_2)
        #aaa
        #sigma_err = torch.sqrt(
        #    sigma_2[0, 0]
        #)  # for now, take [0, 0] component as the obs error
        #print(f"{sigma_err=}")
        h_results = rearrange(h_results_, "(d s st) x -> d s st x", d=N_data, s=N_sample, st=n_step)
        if periodic:
            h_results_periodic = h_results[:, :, :, periodic_indices]
            x_value_periodic = x_value[:, :, periodic_indices]
            h_results_nonperiodic = h_results[:, :, :, nonperiodic_indices]
            x_value_nonperiodic = x_value[:, :, nonperiodic_indices]
            vonmises_dist = dist.von_mises.VonMises(h_results_periodic, concentration_periodic, validate_args=None)
            quad_log_prob_periodic = torch.sum(vonmises_dist.log_prob(x_value_periodic.unsqueeze(1).repeat(1, N_sample, 1, 1)))
            #quad_log_prob_periodic = -0.5 * (periodic_difference(x=h_results_periodic, y=x_value_periodic.unsqueeze(1).repeat(1, N_sample, 1, 1), period=period) / sigma_err)**2
            quad_log_prob_others = -0.5 * ((h_results_nonperiodic-x_value_nonperiodic.unsqueeze(1).repeat(1, N_sample, 1, 1)) / sigma_err)**2
            log_prob_values3 = torch.sum(quad_log_prob_periodic)+torch.sum(quad_log_prob_others)
            #print(f"{torch.sum(quad_log_prob_periodic)=}, {torch.sum(quad_log_prob_others)=}, {log_prob_values3=}")
        else:
            quad_log_prob = -0.5 * ((h_results-x_value.unsqueeze(1).repeat(1, N_sample, 1, 1)) / sigma_err)**2
            log_prob_values3 = torch.sum(quad_log_prob) - N_data*N_sample*n_step*(x_dim/2)*torch.log(sigma_err)
    
            return log_prob_values3 / N_sample / N_data
    else:
        print("not yet implemented")
        pass


def compute_loss_integral2(
    mu_1,
    sigma_1,
    sigma_2,
    h,
    z_dim,
    x_dim,  # obs_dim or physical_dim depending on take_loss_physical
    x_value,
    N_data,
    n_step,
    sigma1_block_diag=False,
    periodic=False,
    periodic_indices=[],
    period=2*torch.pi,
    concentration_periodic=1e4,
):
    #torch.cuda.synchronize()
    #integral2_begin = time.time()
    
    if sigma1_block_diag:
        assert mu_1.shape == (
            N_data,
            n_step,
            z_dim//2,
            2
        ), f"mu_1.shape should be (N_data, n_step, z_dim//2, 2) but is {mu_1.shape}"
        assert sigma_1.shape == (
            N_data,
            n_step,
            z_dim//2,
            2,
            2,
        ), f"sigma_1.shape should be (N_data, n_step, z_dim//2, 2, 2) but is {sigma_1.shape}"
    else:
        assert mu_1.shape == (
            N_data,
            n_step,
            z_dim,
        ), f"mu_1.shape should be (N_data, n_step, z_dim) but is {mu_1.shape}"
        assert sigma_1.shape == (
            N_data,
            n_step,
            z_dim,
            z_dim,
        ), f"sigma_1.shape should be (N_data, n_step, z_dim, z_dim) but is {sigma_1.shape}"
        assert sigma_2.shape == (
            x_dim,
            x_dim,
        ), f"sigma_2.shape should be (x_dim, x_dim) but is {sigma_2.shape}"
        assert x_value.shape == (
            N_data,
            n_step,
            x_dim,
        ), f"x_value.shape should be (N_data, n_step, x_dim) but is {x_value.shape}"
    if periodic:
        nonperiodic_indices = list(set([i for i in range(x_dim)]) - set(periodic_indices))
        #print(f"{periodic_indices=}")
        #print(f"{nonperiodic_indices=}")

    assert ~torch.isnan(sigma_1).any(), f"sigma_1 contains nan: {sigma_1}"
    # sample from N(z;mu_1, sigma_1) and compute
    # \int N(z;mu_1, sigma_1)log[N(x;mu_2(z), sigma_2(z))] dz
    sqrt_sigma_1 = torch.empty_like(sigma_1, device="cuda")
    if sigma1_block_diag:
        matrix = sigma_1.reshape(N_data * n_step, z_dim//2, 2, 2)
    else:
        matrix = sigma_1.reshape(N_data * n_step, z_dim, z_dim)

    #torch.cuda.synchronize()
    #eigendecomposition_begin = time.time()

    if sigma1_block_diag:
        eigenvalues, eigenvectors = torch.linalg.eigh(
            matrix
        )  # (N_data*n_step, z_dim//2, 2), (N_data*n_step, z_dim//2, 2, 2)
        assert ~torch.lt(eigenvalues, 0).any(), eigenvalues
        sqrt_eigenvalues = torch.sqrt(eigenvalues)  # (N_data*n_step, z_dim//2, 2)
        sqrt_matrix_blocks = torch.einsum(
            "ixjl,ixlm->ixjm",
            torch.einsum("ixjk,ixkl->ixjl", eigenvectors, torch.diag_embed(sqrt_eigenvalues)),
            eigenvectors.transpose(2, 3),
        )
        assert ~torch.isnan(sqrt_matrix_blocks).any(), f"{sqrt_matrix_blocks}, {matrix}"
        sqrt_sigma_1_block = sqrt_matrix_blocks.reshape(N_data, n_step, z_dim//2, 2, 2)

        mvn1 = dist.MultivariateNormal(
            torch.zeros(2, device="cuda"), torch.eye(2, device="cuda")
        )
        # N_sample = 5 # for moving MNIST
        #N_sample = 1  # for Lorenz96
        N_sample = 1  # for Lorenz96
        '''
        points = mvn1.sample((N_data, N_sample))
        points_T = (
        points.unsqueeze(1).repeat(1, n_step, 1, 1).transpose(2, 3)
        )  # (N_data, n_step, z_dim, N_sample)
        '''
        points_T = mvn1.sample((N_data, n_step, z_dim//2, N_sample)).transpose(3, 4) # (N_data, n_step, z_dim//2, 2, N_sample)

        #torch.cuda.synchronize()
        #reparametrization_prep_end = time.time()
    else:
        print(f"{matrix.device=}")
        print(f"{matrix.shape=}")
        eigenvalues, eigenvectors = torch.linalg.eigh(
            matrix
        )  # (N_data*n_step, z_dim), (N_data*n_step, z_dim, z_dim)
        assert ~torch.lt(eigenvalues, 0).any(), eigenvalues
        sqrt_eigenvalues = torch.sqrt(eigenvalues)  # (N_data*n_step, z_dim)

        #torch.cuda.synchronize()
        #sqrt_matrix_begin = time.time()

        sqrt_matrix = torch.einsum(
            "ijl,ilm->ijm",
            torch.einsum("ijk,ikl->ijl", eigenvectors, torch.diag_embed(sqrt_eigenvalues)),
            eigenvectors.transpose(1, 2),
        )
        
        assert ~torch.isnan(sqrt_matrix).any(), f"{sqrt_matrix}, {matrix}"
        sqrt_sigma_1 = sqrt_matrix.reshape(N_data, n_step, z_dim, z_dim)

        mvn1 = dist.MultivariateNormal(
            torch.zeros(z_dim, device="cuda"), torch.eye(z_dim, device="cuda")
        )
        # N_sample = 5 # for moving MNIST
        #N_sample = 1  # for Lorenz96
        N_sample = 1  # for Lorenz96
        '''
        points = mvn1.sample((N_data, N_sample))
        points_T = (
        points.unsqueeze(1).repeat(1, n_step, 1, 1).transpose(2, 3)
        )  # (N_data, n_step, z_dim, N_sample)
        '''
        points_T = mvn1.sample((N_data, n_step, N_sample)).transpose(2, 3) # (N_data, n_step, z_dim, N_sample)


        #torch.cuda.synchronize()
        #reparametrization_prep_end = time.time()
    
    if sigma1_block_diag:
        result_block = torch.einsum(
            "abxij,abxik->abxjk", sqrt_sigma_1_block, points_T
        )  # (N_data, n_step, z_dim//2, 2, N_sample)
        #result = rearrange(result_block, "n s z j i -> n s (j z) i")
        result = rearrange(result_block, "n s z j i -> n s (z j) i")
    else:
        result = torch.einsum(
            "abij,abik->abjk", sqrt_sigma_1, points_T
        )  # (N_data, n_step, z_dim, N_sample)

    result = result.transpose(2, 3).transpose(1, 2)  # (N_data, N_sample, n_step, z_dim)

    if sigma1_block_diag:
        mu_1 = rearrange(mu_1, "nd nstep z1 z2 -> nd nstep (z1 z2)")
    h_arg = result + mu_1.unsqueeze(1).repeat(
        1, N_sample, 1, 1
    )  # (N_data, N_sample, n_step, z_dim)

    h_arg = torch.reshape(h_arg, (N_data * N_sample * n_step, z_dim))

    #torch.cuda.synchronize()
    #reparametrization_end = time.time()
    
    h_results_ = h(h_arg)

    #torch.cuda.synchronize()
    #observation_end = time.time()
    
    log_prob_values3 = 0.0
    sigma_err = torch.sqrt(
        sigma_2[0, 0]
    )  # for now, take [0, 0] component as the obs error
    h_results = torch.reshape(h_results_, (N_data, N_sample, n_step, x_dim))
    if periodic:
        h_results_periodic = h_results[:, :, :, periodic_indices]
        x_value_periodic = x_value[:, :, periodic_indices]
        h_results_nonperiodic = h_results[:, :, :, nonperiodic_indices]
        x_value_nonperiodic = x_value[:, :, nonperiodic_indices]
        vonmises_dist = dist.von_mises.VonMises(h_results_periodic, concentration_periodic, validate_args=None)
        quad_log_prob_periodic = torch.sum(vonmises_dist.log_prob(x_value_periodic.unsqueeze(1).repeat(1, N_sample, 1, 1)))
        #quad_log_prob_periodic = -0.5 * (periodic_difference(x=h_results_periodic, y=x_value_periodic.unsqueeze(1).repeat(1, N_sample, 1, 1), period=period) / sigma_err)**2
        quad_log_prob_others = -0.5 * ((h_results_nonperiodic-x_value_nonperiodic.unsqueeze(1).repeat(1, N_sample, 1, 1)) / sigma_err)**2
        log_prob_values3 = torch.sum(quad_log_prob_periodic)+torch.sum(quad_log_prob_others)
        #print(f"{torch.sum(quad_log_prob_periodic)=}, {torch.sum(quad_log_prob_others)=}, {log_prob_values3=}")
    else:
        quad_log_prob = -0.5 * ((h_results-x_value.unsqueeze(1).repeat(1, N_sample, 1, 1)) / sigma_err)**2
        log_prob_values3 = torch.sum(quad_log_prob)
    #mvn2 = dist.normal.Normal(h_results, sigma_err)
    #log_prob_values3 += torch.sum(
    #    mvn2.log_prob(x_value.unsqueeze(1).repeat(1, N_sample, 1, 1))
    #)
    
    #torch.cuda.synchronize()
    #integral2_end = time.time()

    '''
    print(f"prep time: {eigendecomposition_begin-integral2_begin}")
    print(f"eigendecomposition time: {sqrt_matrix_begin-integral2_begin}")
    print(f"sqrt_matrix time: {reparametrization_prep_end-sqrt_matrix_begin}")
    print(f"reparametrization time: {reparametrization_end-reparametrization_prep_end}")
    print(f"observation time: {observation_end-reparametrization_end}")
    print(f"NLL_summation time: {integral2_end-observation_end}")
    '''
    #print(f"loss total time: {integral2_end-integral2_begin}")
    
    
    return log_prob_values3 / N_sample / N_data


def compute_KL_gaussians(mu_1, sigma_1, mu_2, sigma_2, z_dim, N_data, n_step, sigma_block_diag=True):
    if sigma_block_diag:
        assert mu_1.shape == (
            N_data,
            n_step,
            z_dim//2,
            2
        ), f"mu_1.shape should be (N_data, n_step, z_dim//2, 2) but is {mu_1.shape}"
        assert sigma_1.shape == (
            N_data,
            n_step,
            z_dim//2,
            2,
            2,
        ), f"sigma_1.shape should be (N_data, n_step, z_dim//2, 2, 2) but is {sigma_1.shape}"
        assert mu_1.shape == (
            N_data,
            n_step,
            z_dim//2,
            2
        ), f"mu_1.shape should be (N_data, n_step, z_dim//2, 2) but is {mu_1.shape}"
        assert sigma_1.shape == (
            N_data,
            n_step,
            z_dim//2,
            2,
            2,
        ), f"sigma_1.shape should be (N_data, n_step, z_dim//2, 2, 2) but is {sigma_1.shape}"
    else:
        assert mu_1.shape == (
            N_data,
            n_step,
            z_dim,
        ), f"mu_1.shape should be (N_data, n_step, z_dim) but is {mu_1.shape}"
        assert sigma_1.shape == (
            N_data,
            n_step,
            z_dim,
            z_dim,
        ), f"sigma_1.shape should be (N_data, n_step, z_dim, z_dim) but is {sigma_1.shape}"
        assert mu_2.shape == (
            N_data,
            n_step,
            z_dim,
        ), f"mu_2.shape should be (N_data, n_step, z_dim) but is {mu_2.shape}"
        assert sigma_2.shape == (
            N_data,
            n_step,
            z_dim,
            z_dim,
        ), f"sigma_2.shape should be (N_data, n_step, z_dim, z_dim) but is {sigma_2.shape}"
    if sigma_block_diag:
        dists_p = dist.MultivariateNormal(
            mu_1, sigma_1
        )
        dists_q = dist.MultivariateNormal(
            mu_2, sigma_2
        )
        loss_value = torch.sum(dist.kl.kl_divergence(dists_p, dists_q))/N_data
        return loss_value
        '''
        sigma_2inv = torch.inverse(sigma_2)
        loss1 = torch.einsum(
            "ijxkk->ijx", torch.einsum("ijxkl,ijxlm->ijxkm", sigma_2inv, sigma_1)
        ).sum()
        loss2 = torch.einsum(
            "lixk,lixk->l",
            mu_2 - mu_1,
            torch.einsum("lixj,lixjk->lixk", mu_2 - mu_1, sigma_2inv),
        ).sum()
        S1 = torch.linalg.svdvals(sigma_1)
        S2 = torch.linalg.svdvals(sigma_2)
        log_det_sigma_1 = torch.sum(torch.log(S1), 2)
        log_det_sigma_2 = torch.sum(torch.log(S2), 2)
        loss3 = (log_det_sigma_2 - log_det_sigma_1).sum() - z_dim * N_data * n_step
        '''
    else:
        sigma_2inv = torch.inverse(sigma_2)
        loss1 = torch.einsum(
            "ijkk->ij", torch.einsum("ijkl,ijlm->ijkm", sigma_2inv, sigma_1)
        ).sum()
        loss2 = torch.einsum(
            "lik,lik->l",
            mu_2 - mu_1,
            torch.einsum("lij,lijk->lik", mu_2 - mu_1, sigma_2inv),
        ).sum()
        S1 = torch.linalg.svdvals(sigma_1)
        S2 = torch.linalg.svdvals(sigma_2)
        log_det_sigma_1 = torch.sum(torch.log(S1), 2)
        log_det_sigma_2 = torch.sum(torch.log(S2), 2)
        loss3 = (log_det_sigma_2 - log_det_sigma_1).sum() - z_dim * N_data * n_step
        
    assert ~torch.isnan(loss1), loss1
    assert ~torch.isnan(loss2), loss2
    assert ~torch.isnan(loss3), torch.min(torch.det(sigma_1))
    loss_value2 = 0.5 * (loss1 + loss2 + loss3)
    # assert loss_value2 >= 0, loss_value2
    loss_value2 = torch.max(torch.tensor(0.0), loss_value2)
    assert loss_value2 < 1e20, loss_value2
    return loss_value2 / N_data


class DBF_DDP(nn.Module):
    def __init__(
        self,
        mode,
        F,
        h_network,
        take_blockstep,
        take_loss_physical,
        log_sysnoise,
        log_obsnoise,
        G_val,
        z_dim,
        x_dim,
        time_series_input,
        time_series_length,
        aux_alpha,
        save_folder,
        periodic=False,
        periodic_indices=[],
        period=2*torch.pi,
        log_concentration_periodic=None,
        load_model=False,
        load_model_path="",
        variable_kernelsize=False,
        unitmatrix=False,
        sigma_block_diag=True,
        G_nondiag=False,
        kernel=3,
    ):
        super().__init__()
        self.mode = mode
        self.F = F
        self.h_network = h_network
        self.take_blockstep = take_blockstep
        self.take_loss_physical = take_loss_physical
        self.log_sysnoise = torch.tensor(log_sysnoise)
        self.log_obsnoise = torch.nn.Parameter(torch.tensor(log_obsnoise, device="cuda"))
        self.Q = torch.exp(2 * self.log_sysnoise) * torch.eye(z_dim, device="cuda")
        #self.R = torch.exp(2 * self.log_obsnoise) * torch.eye(x_dim, device=device)
        self.G_val = G_val
        self.time_series_input = time_series_input
        if self.time_series_input:
            self.time_series_length = time_series_length
        self.x_dim = x_dim
        self.z_dim = z_dim
        assert (
            self.z_dim % 2 == 0
        ), "z_dim should be even numbered if we do not consider real-valued eigenvalues"
        self.V = 1e8 * torch.eye(self.z_dim).cuda()#to(device)
        self.V_inv = torch.inverse(self.V)
        self.m = 0 * torch.ones(self.z_dim).cuda()#to(device)
        self.u = torch.zeros(self.z_dim).cuda()#to(device)
        self.test_every = 1000
        self.aux_alpha = aux_alpha
        self.save_folder = save_folder
        self.periodic = periodic
        self.periodic_indices = periodic_indices
        self.period = period
        if log_concentration_periodic is not None:
            self.log_concentration_periodic = torch.nn.Parameter(torch.tensor(log_concentration_periodic, device="cuda"))
        else:
            self.log_concentration_periodic = None
        self.load_model = load_model
        self.load_model_path = load_model_path
        self.variable_kernelsize = variable_kernelsize
        self.kernel = kernel
        self.G_nondiag = G_nondiag
        self.network_init(mode=self.mode)
        self.unitmatrix = unitmatrix
        self.sigma_block_diag = sigma_block_diag
        if self.sigma_block_diag:
            self.mu_0 = torch.zeros(2).view(1, 1, 2).repeat(1, self.z_dim//2, 1).cuda()#to(device)
            self.V_init = 100.0 * torch.eye(2).view(1, 1, 2, 2).repeat(1, self.z_dim//2, 1, 1).cuda()#to(device)
            self.V = 1e8 * torch.eye(2).view(1, 2, 2).repeat(self.z_dim//2, 1, 1).cuda()#to(device)
            self.V_inv = torch.inverse(self.V)
            self.m = 0 * torch.ones(2).view(1, 2).repeat(self.z_dim//2, 1).cuda()#to(device)
            self.u = torch.zeros(2).view(1, 2).repeat(self.z_dim//2, 1).cuda()#to(device)
            self.Q = torch.exp(2 * self.log_sysnoise) * torch.eye(2, device="cuda").view(1, 2, 2).repeat(self.z_dim//2, 1, 1)
        else:
            self.mu_0 = torch.zeros(z_dim)
            self.V_init = 100.0 * torch.eye(z_dim).cuda()#to(device)
            self.V = 1e8 * torch.eye(self.z_dim).cuda()#to(device)
            self.V_inv = torch.inverse(self.V)
            self.m = 0 * torch.ones(self.z_dim).cuda()#to(device)
            self.u = torch.zeros(self.z_dim).cuda()#to(device)
            self.Q = torch.exp(2 * self.log_sysnoise) * torch.eye(z_dim, device="cuda")

    def network_init(self, mode):
        assert mode in [
            "Lorenz96_obsspace",
            "Lorenz96_physspace",
            "doublependulum",
            "movinglight1D",
            "movingMNIST2D",
            "movingMNIST3D",
            "movingMNIST2D_modelagnostic",
        ]
        if self.load_model:
            model_wwy = torch.load(self.load_model_path)
            self.f_network = model_wwy.encoder.double()
            self.G_network = model_wwy.encoder.double()
            self.h_network = model_wwy.decoder.double()
            self.lambdas = model_wwy.lambdas
            self.lambdas.requires_grad = False
            self.train_F = False
            Kmatrix = compute_K(
                self.lambdas, steps=2, num_real=0, num_complex=self.z_dim // 2
            )[1].cuda()#to(device)
            self.F = Kmatrix
            print("model loaded")
        elif mode in ["Lorenz96_obsspace", "Lorenz96_physspace"]:
            if self.variable_kernelsize:
                print("variable kernelsize mode")
                (
                    self.f_network,
                    self.G_network,
                    self.h_network,
                    self.lambdas,
                ) = network_init_Lorenz96_kernelsizes(time_sequence_length=self.time_series_length, z_dim=self.z_dim, G_nondiag=self.G_nondiag)
            else:
                (
                    self.f_network,
                    self.G_network,
                    self.h_network,
                    self.lambdas,
                ) = network_init_Lorenz96(time_sequence_length=self.time_series_length, z_dim=self.z_dim, kernel=self.kernel, G_nondiag=self.G_nondiag)
            self.lambdas = nn.Parameter(self.lambdas.cuda())
            self.lambdas.requires_grad = True
            self.train_F = True
        elif mode in ["doublependulum"]:
            (
                self.f_network,
                self.G_network,
                self.h_network,
                self.lambdas,
            ) = network_init_doublependulum(z_dim=self.z_dim, x_dim=self.x_dim)
            self.lambdas.requires_grad = True
            self.train_F = True
        elif mode in ["movinglight1D"]:
            self.f_network, self.G_network = network_init_moving_light_1D(
                z_dim=self.z_dim, x_dim=self.x_dim
            )
            self.train_F = False
            assert self.F is not None
            assert self.h_network is not None
        elif mode in ["movingMNIST2D"]:
            self.f_network, self.G_network = network_init_moving_MNIST_2D(
                z_dim=self.z_dim, x_dim=self.x_dim
            )
            self.train_F = False
            assert self.F is not None
            assert self.h_network is not None
        elif mode in ["movingMNIST3D"]:
            self.f_network, self.G_network = network_init_moving_MNIST_3D(
                z_dim=self.z_dim, x_dim=self.x_dim
            )
            self.train_F = False
            assert self.F is not None
            assert self.h_network is not None
        elif mode in ["movingMNIST2D_modelagnostic"]:
            (
                self.f_network,
                self.G_network,
                self.h_network,
                self.lambdas,
            ) = network_init_moving_MNIST_2D_model_agnostic(
                z_dim=self.z_dim, x_dim=self.x_dim
            )
            self.lambdas.requires_grad = True
            self.train_F = True
        else:
            pass

    def _predict(self, z, V):
        if self.unitmatrix:
            z_new = z
            V_new = V+self.Q
            return z_new, V_new
        if self.sigma_block_diag: # self.F.shape == z_dim//2, 2, 2
            z_new = torch.einsum("zij,bzj->bzi", self.F, z) + self.u
            V_new = torch.einsum("bzik,zlk->bzil", torch.einsum("zij,bzjk->bzik", self.F, V), self.F) + self.Q
        else:
            z_new = torch.matmul(self.F, z.T).T + self.u
            V_new = torch.matmul(torch.matmul(self.F, V), self.F.T) + self.Q
        return z_new, V_new

    def _backward(self, zjp1, zj_forward, Vjp1, Vj_forward):
        Pj = torch.matmul(torch.matmul(self.F, Vj_forward), self.F.T) + self.Q
        Pj_inv = torch.inverse(Pj)
        Cj = torch.matmul(torch.matmul(Vj_forward, self.F.T), Pj_inv)
        zj = zj_forward + torch.einsum(
            "bij,bj->bi", Cj, (zjp1 - torch.matmul(self.F, zj_forward.T).T - self.u)
        )
        Vj = Vj_forward + torch.einsum(
            "bij,bjk->bik", torch.matmul(Cj, (Vjp1 - Pj)), Cj.transpose(1, 2)
        )
        return zj, Vj

    def compute_f_and_G(
        self,
        o_t,
    ):
        N_data = len(o_t)
        if self.sigma_block_diag:
            #print(f"{o_t.device=}")
            f_array = self.f_network(o_t).view(
                N_data, self.z_dim//2, 2
            )  # accept N_data observation
            G_squared = rearrange(self.G_network(o_t) * self.G_network(o_t), "d (z1 z2) -> d z1 z2", z1=self.z_dim//2, z2=2) # (N_data, z_dim//2, 2)
            #G_matrix = self.G_val * torch.diag_embed(G_squared) + 1e-6 * torch.eye(
            #    self.z_dim
            #).to(device)
            G_matrix = self.G_val * torch.diag_embed(G_squared).cuda()#to(device)
            #print(f"{G_matrix=}")
            #print(f"{G_squared[50,:2]=}")
        else:
            f_array = self.f_network(o_t).view(
                N_data, self.z_dim
            )  # accept N_data observation
            if self.G_nondiag:
                G_matrix_ = torch.zeros(N_data, self.z_dim, self.z_dim)
                G_network_output = self.G_network(o_t) # (N_data, z_dim)
                for i in range(self.z_dim//2):
                    a = G_network_output[:, 3*i]
                    b = G_network_output[:, 3*i+1]
                    c = G_network_output[:, 3*i+2]
                    #print(f"{a.shape=}, {b.shape=}, {c.shape=}")
                    G_matrix_[:, 2*i, 2*i] = a*a
                    G_matrix_[:, 2*i, 2*i+1] = a*b
                    G_matrix_[:, 2*i+1, 2*i] = a*b
                    G_matrix_[:, 2*i+1, 2*i+1] = a*b+c*c
                    
                #G_matrix = self.G_val * G_matrix_ + 1e-6 * torch.eye(
                #    self.z_dim
                #).to(device)
                G_matrix = self.G_val * G_matrix_.cuda()#to(device)
            else:
                G_squared = self.G_network(o_t) * self.G_network(o_t)  # (N_data, z_dim)
                #G_matrix = self.G_val * torch.diag_embed(G_squared) + 1e-6 * torch.eye(
                #    self.z_dim
                #).to(device)
                G_matrix = self.G_val * torch.diag_embed(G_squared).cuda()#to(device)
                #print(f"{G_matrix=}")
                #print(f"{G_squared[50,:5]=}")
        G_matrix_inv = torch.inverse(G_matrix) + self.V_inv
        #print(f"{G_matrix_inv[50,0,0]=}, {G_matrix_inv[50,1,1]=}, {G_matrix_inv[50,2,2]=}")
        assert ~torch.isnan(
            G_matrix_inv
        ).any(), f"o_t: {o_t}, G_matrix: {G_matrix}, f_array: {f_array}"
        return f_array, G_matrix_inv

    def _compute_mu_t_sigma_t(self, obs_data, n_step, N_data, jump_step):
        if self.time_series_input:
            assert obs_data.shape == (
                N_data,
                n_step,
                self.time_series_length,
                self.x_dim,
            ), f"obs_data.shape should be (N_data={N_data}, n_step={n_step}, time_series_length={self.time_series_length}, x_dim={self.x_dim})\
 but is {obs_data.shape}"
        else:
            assert obs_data.shape == (
                N_data,
                n_step,
                self.x_dim,
            ), f"obs_data.shape should be (N_data={N_data}, n_step={n_step}, x_dim={self.x_dim}) but is {obs_data.shape}"
        #assert n_step % jump_step == 0

        mu_t_list_all = []
        mu_t_p_list_all = []
        sigma_t_list_all = []
        sigma_t_p_list_all = []
        if self.sigma_block_diag:
            mu_t_p = self.mu_0.repeat(N_data, 1, 1)
            sigma_t_p = self.V_init.repeat(N_data, 1, 1, 1)
            '''
            mu_t_p = (
                mu_0.unsqueeze(0).repeat(N_data, 1, 1, 1).squeeze(1)
            )  # (self.batch_size, z_dim)
            sigma_t_p = (
                sigma_0.unsqueeze(0).repeat(N_data, 1, 1, 1, 1).squeeze(1)
            )  # (self.batch_size, z_dim, z_dim)
            '''
            mu_t_p_list_all.append(mu_t_p)
            sigma_t_p_list_all.append(sigma_t_p)
        else:
            mu_0 = self.mu_0.view(1, self.z_dim).cuda()#to(device)
            sigma_0 = self.V_init.view(1, self.z_dim, self.z_dim).cuda()#to(device)
            mu_t_p = (
                mu_0.unsqueeze(0).repeat(N_data, 1, 1).squeeze(1)
            )  # (self.batch_size, z_dim)
            sigma_t_p = (
                sigma_0.unsqueeze(0).repeat(N_data, 1, 1, 1).squeeze(1)
            )  # (self.batch_size, z_dim, z_dim)
            mu_t_p_list_all.append(mu_t_p)
            sigma_t_p_list_all.append(sigma_t_p)

        if self.time_series_input:
            o_t_all = rearrange(obs_data, "bs nstep tseq xdim -> (bs nstep) tseq xdim")
        else:
            o_t_all = rearrange(obs_data, "bs nstep xdim -> (bs nstep) xdim")

        ft_all, Gt_inv_all = self.compute_f_and_G(
            o_t=o_t_all,
        )
        if torch.isnan(Gt_inv_all).any():
            return None, None, None, None

        if self.sigma_block_diag:
            ft_all_reshaped = rearrange(ft_all, "(bs nstep) z1 z2 -> bs nstep z1 z2", bs=N_data, nstep=n_step)
            Gt_inv_all_reshaped = rearrange(Gt_inv_all, "(bs nstep) z1 z2 z3 -> bs nstep z1 z2 z3", bs=N_data, nstep=n_step)#Gt_inv_all.view(N_data, n_step, self.z_dim//2, 2, 2)
            #print(f"{Gt_inv_all_reshaped[0, 0, 0:2]=}")
            ftGt = torch.einsum("dsabc,dsac->dsab", Gt_inv_all_reshaped, ft_all_reshaped)
        else:
            ft_all_reshaped = ft_all.view(N_data, n_step, self.z_dim)
            Gt_inv_all_reshaped = Gt_inv_all.view(N_data, n_step, self.z_dim, self.z_dim)

        if self.sigma_block_diag: # self.F.shape == z_dim//2, 2, 2
            Kmatrix = compute_K_sigma_block_diag(
                self.lambdas, steps=2, num_real=0, num_complex=self.z_dim // 2
            ).cuda()#to(device)
            self.F = Kmatrix
        elif self.train_F:
            Kmatrix = compute_K(
                self.lambdas, steps=2, num_real=0, num_complex=self.z_dim // 2
            )[1].cuda()#to(device)
            self.F = Kmatrix

        for i in range(n_step):
            # mu_t_p: N_data, zdim//2, 2
            # sigma_t_p: N_data, zdim//2, 2, 2
            #ft_all_reshaped: N_data, n_step, z_dim//2, 2
            #Gt_inv_all_reshaped: N_data, n_step, z_dim//2, 2, 2
            if i % jump_step == 0:
                
                if self.sigma_block_diag:
                    #Gt_inv = Gt_inv_all_reshaped[:, i]  # (self.batch_size, z_dim//2, 2, 2)
                    #ft = ft_all_reshaped[:, i]  # (self.batch_size, z_dim//2, 2)
                    sigma_t_p_inv = torch.inverse(sigma_t_p)  # (self.batch_size, z_dim//2, 2, 2)
                    sigma_t = torch.inverse(sigma_t_p_inv + Gt_inv_all_reshaped[:, i] - self.V_inv) # (self.batch_size, z_dim//2, 2, 2)
                    # sigma_t = sigma_t  # + 1e-4*torch.eye(self.z_dim).to(device) # to stabilize
                    weighted_sum = (
                        torch.einsum("sabc,sac->sab", sigma_t_p_inv, mu_t_p)
                        + ftGt[:, i]
                        - torch.einsum("abc,ac->ab", self.V_inv, self.m)
                    )
                    mu_t = torch.einsum("sabc,sac->sab", sigma_t, weighted_sum)
                    #image = torchviz.make_dot(mu_t_p, params=dict(self.f_network.named_parameters())).render(f"mu_t_p_step{i}", format="png")
                    #image = torchviz.make_dot(mu_t, params=dict(self.f_network.named_parameters())).render(f"mu_t_step{i}", format="png")
                    #image = torchviz.make_dot(sigma_t, params=dict(self.f_network.named_parameters())).render(f"sigma_t_step{i}", format="png")
                    #image = torchviz.make_dot(sigma_t_p, params=dict(self.f_network.named_parameters())).render(f"sigma_t_p_step{i}", format="png")
                    #mu_t = torch.matmul(sigma_t, weighted_sum.unsqueeze(-1)).squeeze(-1)
                    sigma_t_list_all.append(sigma_t)  # (self.batch_size, z_dim//2, 2, 2)
                    mu_t_list_all.append(mu_t)  # (self.batch_size, z_dim//2, 2)
                    assert ~torch.isnan(
                        sigma_t
                    ).any(), (
                        f"sigma_t: {sigma_t}, Gt_inv: {Gt_inv}, sigma_t_p_inv: {sigma_t_p_inv}"
                    )
                else:
                    Gt_inv = Gt_inv_all_reshaped[:, i]  # (self.batch_size, z_dim, z_dim)
                    ft = ft_all_reshaped[:, i]  # (self.batch_size, z_dim)
                    sigma_t_p_inv = torch.inverse(sigma_t_p)  # (self.batch_size, z_dim, z_dim)
                    sigma_t = torch.inverse(sigma_t_p_inv + Gt_inv - self.V_inv)
                    # sigma_t = sigma_t  # + 1e-4*torch.eye(self.z_dim).to(device) # to stabilize
                    weighted_sum = (
                        torch.matmul(sigma_t_p_inv, mu_t_p.unsqueeze(-1)).squeeze(-1)
                        + torch.matmul(Gt_inv, ft.unsqueeze(-1)).squeeze(-1)
                        - torch.matmul(self.V_inv, self.m)
                    )
                    mu_t = torch.matmul(sigma_t, weighted_sum.unsqueeze(-1)).squeeze(-1)
                    sigma_t_list_all.append(sigma_t)  # (self.batch_size, z_dim, z_dim)
                    mu_t_list_all.append(mu_t)  # (self.batch_size, z_dim)
                    assert ~torch.isnan(
                        sigma_t
                    ).any(), (
                        f"sigma_t: {sigma_t}, Gt_inv: {Gt_inv}, sigma_t_p_inv: {sigma_t_p_inv}"
                    )
            else:
                mu_t = mu_t_p
                sigma_t = sigma_t_p
                mu_t_list_all.append(mu_t)
                sigma_t_list_all.append(sigma_t)
                #print(f"skipped, {i=}, {jump_step=}")
            if i < n_step - 1:
                mu_t_p, sigma_t_p = self._predict(mu_t, sigma_t)
                sigma_t_p = sigma_t_p
                sigma_t_p_inv = torch.inverse(sigma_t_p)
                sigma_t_p_list_all.append(sigma_t_p)  # (self.batch_size, z_dim, z_dim)
                mu_t_p_list_all.append(mu_t_p)  # (self.batch_size, z_dim)

        mu_t_list_all = torch.stack(mu_t_list_all).transpose(0, 1)
        mu_t_p_list_all = torch.stack(mu_t_p_list_all).transpose(0, 1)
        sigma_t_list_all = torch.stack(sigma_t_list_all).transpose(0, 1)
        sigma_t_p_list_all = torch.stack(sigma_t_p_list_all).transpose(0, 1)
        # print(f"{mu_t_list_all=}")
        return mu_t_p_list_all, sigma_t_p_list_all, mu_t_list_all, sigma_t_list_all

    def _compute_mu_t_sigma_t_blockstep(self, obs_data, n_step, block_step, N_data=2):
        assert n_step % block_step == 0, f"{n_step=}, {block_step=}"
        num_blocks_n = n_step // block_step

        (
            mu_t_p_list_all,
            sigma_t_p_list_all,
            mu_t_list_all,
            sigma_t_list_all,
        ) = self._compute_mu_t_sigma_t(obs_data, n_step, N_data=N_data)

        # mu_t_backward_predict_all = torch.tensor([], device=device) # store predictions starting from backward results.
        # sigma_t_backward_predict_all = torch.tensor([], device=device) # store predictions starting from backward results.
        mu_t_backward_all = torch.tensor([], device="cuda")  # store backward results.
        sigma_t_backward_all = torch.tensor(
            [], device="cuda"
        )  # store backward results.

        for block in range(num_blocks_n):
            mu_t_list_assimilate_backward_thisblock = []
            sigma_t_list_assimilate_backward_thisblock = []

            final_step = (block + 1) * block_step
            mu_t_hat = mu_t_list_all[:, final_step - 1]  # p(z_t|x_1:t)
            sigma_t_hat = sigma_t_list_all[:, final_step - 1]
            mu_t_list_assimilate_backward_thisblock.append(mu_t_hat)
            sigma_t_list_assimilate_backward_thisblock.append(sigma_t_hat)

            for i in range(1, block_step):  # backward computation
                mu_t_forward = mu_t_list_all[:, final_step - i - 1]
                sigma_t_forward = sigma_t_list_all[:, final_step - i - 1]
                mu_t_hat, sigma_t_hat = self._backward(
                    zjp1=mu_t_hat,
                    zj_forward=mu_t_forward,
                    Vjp1=sigma_t_hat,
                    Vj_forward=sigma_t_forward,
                )
                mu_t_list_assimilate_backward_thisblock.append(mu_t_hat)
                sigma_t_list_assimilate_backward_thisblock.append(sigma_t_hat)

            mu_t_list_assimilate_backward_thisblock = torch.stack(
                mu_t_list_assimilate_backward_thisblock
            ).transpose(0, 1)
            sigma_t_list_assimilate_backward_thisblock = torch.stack(
                sigma_t_list_assimilate_backward_thisblock
            ).transpose(0, 1)

            # mu_t_backward_predict_thisblock = torch.stack(mu_t_backward_predict_thisblock).transpose(0, 1)
            # sigma_t_backward_predict_thisblock = torch.stack(sigma_t_backward_predict_thisblock).transpose(0, 1)
            mu_t_assimilate_backward_thisblock = torch.flip(
                mu_t_list_assimilate_backward_thisblock, dims=[1]
            )
            sigma_t_assimilate_backward_thisblock = torch.flip(
                sigma_t_list_assimilate_backward_thisblock, dims=[1]
            )

            # mu_t_backward_predict_all = torch.cat([mu_t_backward_predict_all, mu_t_backward_predict_thisblock], dim=1)
            # sigma_t_backward_predict_all = torch.cat([sigma_t_backward_predict_all, sigma_t_backward_predict_thisblock], dim=1)
            mu_t_backward_all = torch.cat(
                [mu_t_backward_all, mu_t_assimilate_backward_thisblock], dim=1
            )
            sigma_t_backward_all = torch.cat(
                [sigma_t_backward_all, sigma_t_assimilate_backward_thisblock], dim=1
            )

        return (
            mu_t_p_list_all,
            sigma_t_p_list_all,
            mu_t_list_all,
            sigma_t_list_all,
            mu_t_backward_all,
            sigma_t_backward_all,
        )

    def predict_msteps(self, m_step, mu_t_init, sigma_t_init):
        mu_t_pure_predict = []
        sigma_t_pure_predict = []
        mu_t = mu_t_init.detach()
        sigma_t = sigma_t_init.detach()
        for i in range(m_step):  # pure predict computation
            mu_t, sigma_t = self._predict(mu_t, sigma_t)  # p(z_t+i|x_1:t)
            sigma_t_pure_predict.append(sigma_t)  # (self.batch_size, z_dim, z_dim)
            mu_t_pure_predict.append(mu_t)  # (self.batch_size, z_dim)
        mu_t_pure_predict = torch.stack(mu_t_pure_predict)
        sigma_t_pure_predict = torch.stack(sigma_t_pure_predict)
        return mu_t_pure_predict, sigma_t_pure_predict

    #def compute_predictions(self, obs_data, n_step, m_step, block_step, jump_step):
    def forward(self, obs_data, n_step, m_step, block_step, jump_step, return_concentration=False):
        if self.take_blockstep:
            (
                mu_t_p_list_all,
                sigma_t_p_list_all,
                mu_t_list_all,
                sigma_t_list_all,
                mu_t_backward_all,
                sigma_t_backward_all,
            ) = self._compute_mu_t_sigma_t_blockstep(
                obs_data[:, : n_step + m_step],
                n_step + m_step,
                block_step=block_step,
                N_data=len(obs_data),
                jump_step=jump_step,
            )
            mu_t_backward_all_reshaped = rearrange(
                mu_t_backward_all, "bs s z -> (bs s) z"
            )
            h_output_filtered_reshaped = self.h_network(mu_t_backward_all_reshaped)
            h_output_filtered = rearrange(
                h_output_filtered_reshaped,
                "(bs s) x -> bs s x",
                bs=len(mu_t_backward_all),
                s=n_step + m_step,
            )
        else:
            (
                mu_t_p_list_all,
                sigma_t_p_list_all,
                mu_t_list_all,
                sigma_t_list_all,
            ) = self._compute_mu_t_sigma_t(
                obs_data[:, : n_step + m_step], n_step + m_step, N_data=len(obs_data), jump_step=jump_step
            ) # if sigma_block_diag, mu_t_p_list_all.shape = N_data, n_step+m_step, z_dim//2, 2
            if self.sigma_block_diag:
                mu_t_list_all_reshaped = rearrange(mu_t_list_all, "bs s z1 z2 -> (bs s) (z1 z2)")
            else:
                mu_t_list_all_reshaped = rearrange(mu_t_list_all, "bs s z -> (bs s) z")
            h_output_filtered_reshaped = self.h_network(mu_t_list_all_reshaped)
            h_output_filtered = rearrange(
                h_output_filtered_reshaped,
                "(bs s) x -> bs s x",
                bs=len(mu_t_list_all),
                s=n_step + m_step,
            )
            mu_t_backward_all = None
            sigma_t_backward_all = None

        # long time prediction for auxiliary loss computation
        if m_step > 0:
            if self.take_blockstep:
                mu_t_init = mu_t_backward_all[:, n_step - 1]
                sigma_t_init = sigma_t_backward_all[:, n_step - 1]
            else:
                mu_t_init = mu_t_list_all[:, n_step - 1]
                sigma_t_init = sigma_t_list_all[:, n_step - 1]
            mu_t_predict, sigma_t_predict = self.predict_msteps(
                m_step, mu_t_init, sigma_t_init
            )
            mu_t_predict = torch.cat(
                [mu_t_p_list_all[:, :n_step], mu_t_predict.transpose(0, 1)], dim=1
            )  # (bs, s, z1, z2)
            # print(f"{mu_t_predict.shape=}")
            sigma_t_predict = torch.cat(
                [sigma_t_p_list_all[:, :n_step], sigma_t_predict.transpose(0, 1)], dim=1
            )  # (bs, s, z1, z2, z3)
            
            #print(f"{mu_t_predict.shape=}")
            #print(f"{sigma_t_predict.shape=}")
            mu_t_predict_reshaped = rearrange(mu_t_predict, "bs s z1 z2 -> (bs s) (z1 z2)")
        else:
            if self.sigma_block_diag:
                mu_t_predict = 0
                mu_t_predict_reshaped = rearrange(mu_t_p_list_all, "bs s z1 z2 -> (bs s) (z1 z2)")
            else:
                mu_t_predict = 0
                mu_t_predict_reshaped = rearrange(mu_t_p_list_all, "bs s z -> (bs s) z")

        h_output_reshaped = self.h_network(mu_t_predict_reshaped)
        h_output = rearrange(
            h_output_reshaped,
            "(bs s) x -> bs s x",
            bs=len(mu_t_list_all),
            s=n_step + m_step,
        )
        #mu_t_predict = mu_t_predict_reshaped
        # print(f"{h_output.shape=}")
        h_results = self.reparametrize(mu_t_list_all, sigma_t_list_all)

        if return_concentration:
            return (
                mu_t_p_list_all,
                sigma_t_p_list_all,
                mu_t_list_all,
                sigma_t_list_all,
                mu_t_predict,
                sigma_t_predict,
                h_output,
                h_output_filtered,
                h_results,
                self.log_obsnoise,
                self.log_concentration_periodic,
            )
        else:
            return (
                mu_t_p_list_all,
                sigma_t_p_list_all,
                mu_t_list_all,
                sigma_t_list_all,
                mu_t_predict,
                sigma_t_backward_all,
                h_output,
                h_output_filtered,
                h_results,
                self.log_obsnoise,
            )

    def reparametrize(self, mu_1, sigma_1): # works only for sigma_block_diag==True
        (N_data, n_step, z_dim_1, z_dim_2) = mu_1.shape
        z_dim = z_dim_1*z_dim_2
        sqrt_sigma_1 = torch.linalg.cholesky(sigma_1)
        mvn1 = dist.Normal(torch.tensor(0.0, device="cuda"), torch.tensor(1.0, device="cuda"))
        N_sample = 1
        points = mvn1.sample((N_data, n_step, N_sample, z_dim//2, 2))
        result = mu_1.unsqueeze(2).repeat(1, 1, N_sample, 1, 1) + torch.einsum(
            "abxij,absxj->absxi", sqrt_sigma_1, points
        )  # (N_data, n_step, N_sample, z_dim//2, 2)
        h_arg = result.transpose(1, 2) # (N_data, N_sample, n_step, z_dim//2, 2)
        h_arg = rearrange(h_arg, "d s st z1 z2 -> (d s st) (z1 z2)")
        h_results_ = self.h_network(h_arg)
        h_results = rearrange(h_results_, "(d s st) x -> d s st x", d=N_data, s=N_sample, st=n_step)
        return h_results

    '''
    #def compute_loss(self, obs_data, phys_target, n_step, m_step, block_step, jump_step, jump_step_loss=1, noKL=False):
    def forward(self, obs_data, phys_target, n_step, m_step, block_step, jump_step, jump_step_loss=1, noKL=False):
        #torch.cuda.synchronize()
        #loss_begin = time.time()
        if self.take_loss_physical:
            if self.time_series_input:
                x_val = phys_target[:, :, -1, :]
            else:
                x_val = phys_target[:, :, :]
        else:
            if self.time_series_input:
                x_val = obs_data[:, :, -1, :]
            else:
                x_val = obs_data[:, :, :]
        #torch.cuda.synchronize()
        #predict_begin = time.time()

        (
            mu_t_p_list_all,
            sigma_t_p_list_all,
            mu_t_list_all,
            sigma_t_list_all,
            mu_t_backward_all,
            sigma_t_backward_all,
            h_output,
            h_output_filtered,
        ) = self.compute_predictions(
            obs_data[:, : n_step + m_step], n_step, m_step, block_step, jump_step=jump_step
        )
        #print(f"{sigma_t_p_list_all[0, 0:3, 0]=}")
        #print(f"{sigma_t_list_all[0, 0:3, 0]=}")
        # print(f"{h_output[0]=}")
        # print(f"{x_val[0]=}")
        pred_simple2 = F.mse_loss(h_output, x_val)
        
        #torch.cuda.synchronize()
        #loss_integral_begin = time.time()
        
        if self.take_blockstep:
            #print(f"{int(np.ceil((n_step+m_step)/jump_step_loss))=}")
            loss_integral = compute_loss_integral2(
                mu_1=mu_t_backward_all[:, ::jump_step_loss, :],
                sigma_1=sigma_t_backward_all[:, ::jump_step_loss, :, :],
                sigma_2=self.R,
                h=self.h_network,
                z_dim=self.z_dim,
                x_dim=self.x_dim,
                x_value=x_val[:, ::jump_step_loss, :],
                N_data=len(x_val),
                n_step=int(np.ceil((n_step + m_step)/jump_step_loss)),
                periodic=self.periodic,
                periodic_indices=self.periodic_indices,
                period=self.period,
                concentration=self.concentration_periodic,
                sigma1_block_diag=self.sigma_block_diag,
                # train_1shape=train_1shape,
                # train_2shapes=train_2shapes,
                # shape_torch1=shape_torch1,
                # shape_torch2=shape_torch2,
            )
            loss_KLp = compute_KL_gaussians(
                mu_1=mu_t_backward_all[:, ::jump_step_loss, :],
                sigma_1=sigma_t_backward_all[:, ::jump_step_loss, :, :],
                mu_2=mu_t_p_list_all[:, ::jump_step_loss, :],
                sigma_2=sigma_t_p_list_all[:, ::jump_step_loss, :, :],
                z_dim=self.z_dim,
                N_data=len(x_val),
                n_step=np.ceil((n_step + m_step)/jump_step_loss),
                sigma_block_diag=self.sigma_block_diag,
            )
            loss_KLf = compute_KL_gaussians(
                mu_1=mu_t_backward_all[:, ::jump_step_loss, :],
                sigma_1=sigma_t_backward_all[:, ::jump_step_loss, :, :],
                mu_2=mu_t_list_all[:, ::jump_step_loss, :],
                sigma_2=sigma_t_list_all[:, ::jump_step_loss, :, :],
                z_dim=self.z_dim,
                N_data=len(x_val),
                n_step=np.ceil((n_step + m_step)/jump_step_loss),
                sigma_block_diag=self.sigma_block_diag,
            )
            # print(f"{loss_KLp=}")
            # print(f"{loss_KLf=}")
            loss_KL = loss_KLp - loss_KLf
        else:
            #print(f"{int(np.ceil((n_step+m_step)/jump_step_loss))=}")
            #loss_integral = compute_loss_integral2(
            loss_integral = compute_loss_integral3(
                mu_1=mu_t_list_all[:, ::jump_step_loss, :],
                sigma_1=sigma_t_list_all[:, ::jump_step_loss, :, :],
                sigma_2=self.log_obsnoise,
                h=self.h_network,
                z_dim=self.z_dim,
                x_dim=self.x_dim,
                x_value=x_val[:, ::jump_step_loss, :],
                N_data=len(x_val),
                n_step=int(np.ceil((n_step + m_step)/jump_step_loss)),
                periodic=self.periodic,
                periodic_indices=self.periodic_indices,
                period=self.period,
                concentration_periodic=self.concentration_periodic,
                sigma1_block_diag=self.sigma_block_diag,
                # train_1shape=train_1shape,
                # train_2shapes=train_2shapes,
                # shape_torch1=shape_torch1,
                # shape_torch2=shape_torch2,
            )

            #torch.cuda.synchronize()
            #loss_KL_begin = time.time()
            
            loss_KL = compute_KL_gaussians(
                mu_1=mu_t_list_all[:, ::jump_step_loss, :],
                sigma_1=sigma_t_list_all[:, ::jump_step_loss, :, :],
                mu_2=mu_t_p_list_all[:, ::jump_step_loss, :],
                sigma_2=sigma_t_p_list_all[:, ::jump_step_loss, :, :],
                z_dim=self.z_dim,
                N_data=len(x_val),
                n_step=np.ceil((n_step + m_step)/jump_step_loss),
                sigma_block_diag=self.sigma_block_diag,
            )
            # print(f"{loss_KL=}")
        if noKL:
            loss = -loss_integral + self.aux_alpha * pred_simple2
        else:
            loss = loss_KL - loss_integral + self.aux_alpha * pred_simple2

        #torch.cuda.synchronize()
        #loss_end = time.time()
        #print(f"loss_KL time: {loss_end-loss_KL_begin}")
        #print(f"loss_integral time: {loss_KL_begin-loss_integral_begin}")
        #print(f"compute_predict time: {loss_integral_begin-predict_begin}")
        #print(f"loss total time: {loss_end-loss_begin}")
        
        return loss, loss_integral, loss_KL, pred_simple2
    '''

    '''
    def train(
        self,
        num_epochs,
        dataloader,
        testloader,
        train_var,
        lr,
        n_step,
        m_step,
        n_step_test,
        m_step_test,
        block_step=0,
        jump_step=1, # 1 for no jumping
        jump_step_loss=1, # 1 for no jumping
        noKL=False,
        gamma_per_iter=1,
    ):
        if self.take_blockstep:
            assert block_step > 0

        assert train_var in ["f", "fG", "fhK", "fhKG", "fhKGR", "fhKGQ","fhKGRQ"]
        print(f"{train_var=}")
        if train_var == "f":
            parameters = [{"params": self.f_network.parameters()}]
        elif train_var == "fG":
            parameters = [{"params": self.f_network.parameters()}]
            parameters = [{"params": self.G_network.parameters()}]
        elif train_var == "fhK":
            parameters = [
                {"params": self.f_network.parameters()},
                {"params": self.h_network.parameters()},
                {"params": self.lambdas},
            ]
        elif train_var == "fhKG":
            parameters = [
                {"params": self.f_network.parameters()},
                {"params": self.G_network.parameters()},
                {"params": self.h_network.parameters()},
                {"params": self.lambdas},
            ]
        elif train_var == "fhKGR":
            self.log_obsnoise.requires_grad = True
            parameters = [
                {"params": self.f_network.parameters()},
                {"params": self.G_network.parameters()},
                {"params": self.h_network.parameters()},
                {"params": self.lambdas},
                {"params": self.log_obsnoise},
            ]
        elif train_var == "fhKGQ":
            self.log_sysnoise.requires_grad = True
            parameters = [
                {"params": self.f_network.parameters()},
                {"params": self.G_network.parameters()},
                {"params": self.h_network.parameters()},
                {"params": self.lambdas},
                {"params": self.log_sysnoise},
            ]
        elif train_var == "fhKGRQ":
            self.log_obsnoise.requires_grad = True
            self.log_sysnoise.requires_grad = True
            parameters = [
                {"params": self.f_network.parameters()},
                {"params": self.G_network.parameters()},
                {"params": self.h_network.parameters()},
                {"params": self.lambdas},
                {"params": self.log_obsnoise},
                {"params": self.log_sysnoise},
            ]

        optimizer = torch.optim.Adam(parameters, lr=lr)
        scheduler = ExponentialLR(optimizer, gamma=gamma_per_iter)
        for epoch in range(num_epochs):
            for i_iter, batch in enumerate(dataloader):
                if i_iter % self.test_every == 0:
                    print(f"{i_iter=}")
                    testloss, testloss_integral, testloss_KL, testloss_aux = self.test(
                        testloader, n_step_test, m_step_test, block_step, jump_step, jump_step_loss, noKL=noKL, savename="last",#savename=f"{i_iter:05d}",
                    )
                #torch.cuda.synchronize()
                #iter_begin = time.time()
                if self.take_loss_physical:
                    obs_data = batch[0]
                    phys_data = batch[1]
                else:
                    obs_data = batch[0]
                    phys_data = None
                optimizer.zero_grad(set_to_none=True)
                loss, loss_integral, loss_KL, loss_aux = self.compute_loss(
                    obs_data, phys_data, n_step, m_step, block_step, jump_step, jump_step_loss, noKL=noKL,
                )
                if i_iter % 20 == 0:
                    with open(os.path.join(self.save_folder, "trainloss_integral.txt"), "a") as f:
                        f.write(f"{-loss_integral.detach().cpu().numpy()}\n")
                    with open(os.path.join(self.save_folder, "trainloss_KL.txt"), "a") as f:
                        f.write(f"{loss_KL.detach().cpu().numpy()}\n")
                    with open(os.path.join(self.save_folder, "log_obsnoise.txt"), "a") as f:
                        f.write(f"{self.log_obsnoise.detach().cpu().numpy()}\n")
                    with open(os.path.join(self.save_folder, "log_sysnoise.txt"), "a") as f:
                        f.write(f"{self.log_sysnoise.detach().cpu().numpy()}\n")
                    print(f"{loss_aux=}, {self.aux_alpha=}")
                    print(f"{loss=}, {loss_integral=}, {loss_KL=}")
                    print(f"{self.log_obsnoise=}")
                    print(f"{self.log_sysnoise=}")

                #x=torch.ones(10, requires_grad=True)
                #weights = {'x':x}
                
                #y=x**2
                #z=x**3
                #r=(y+z).sum()
                #torchviz.make_dot(r).render("attached", format="png")
                
                #image = torchviz.make_dot(loss, params=dict(self.f_network.named_parameters())).render("loss", format="png")
                #image = torchviz.make_dot(loss_KL, params=dict(self.f_network.named_parameters())).render("loss_KL", format="png")
                #image = torchviz.make_dot(loss_integral, params=dict(self.f_network.named_parameters())).render("loss_integral", format="png")
                #aaa
                #torch.cuda.synchronize()
                #loss_backward_begin = time.time()
                loss.backward()
                #torch.cuda.synchronize()
                #loss_backward_end = time.time()
                
                #print(f"{loss_backward_end-loss_backward_begin=}")
                
                optimizer.step()
                scheduler.step()
                print(f"{scheduler.get_last_lr()=}")

                if train_var in ["fhKGQ", "fhKGRQ"]:
                    self.Q = torch.exp(2 * self.log_sysnoise) * torch.eye(2, device=device).view(1, 2, 2).repeat(self.z_dim//2, 1, 1)
                #torch.cuda.synchronize()
                #iter_end = time.time()
                
                #print(f"{iter_end-iter_begin=}")

    def test(self, testloader, n_step_test, m_step_test, block_step, jump_step, jump_step_loss, noKL=False, savename=""):
        with torch.no_grad():
            for i_iter, batch in enumerate(testloader):
                if self.take_loss_physical:
                    obs_data = batch[0]
                    phys_data = batch[1]
                else:
                    obs_data = batch[0]
                    phys_data = None
                loss, loss_integral, loss_KL, loss_aux = self.compute_loss(
                    obs_data, phys_data, n_step_test, m_step_test, block_step, jump_step, jump_step_loss, noKL=noKL
                )
                print("test set")
                with open(os.path.join(self.save_folder, "testloss_integral.txt"), "a") as f:
                    f.write(f"{loss_integral.detach().cpu().numpy()}\n")
                with open(os.path.join(self.save_folder, "testloss_KL.txt"), "a") as f:
                    f.write(f"{loss_KL.detach().cpu().numpy()}\n")

                print(f"{loss_aux=}, {self.aux_alpha=}")
                print(f"{loss=}, {loss_integral=}, {loss_KL=}")

                # data output
                
                (
                    mu_t_p_list_all,
                    sigma_t_p_list_all,
                    mu_t_list_all,
                    sigma_t_list_all,
                    mu_t_backward_all,
                    sigma_t_backward_all,
                    h_output,
                    h_output_filtered
                )= self.compute_predictions(
                    obs_data[:, : n_step_test + m_step_test],
                    n_step_test,
                    m_step_test,
                    block_step,
                    jump_step,
                )
                if self.take_loss_physical:
                    if self.time_series_input:
                        x_val = phys_data[:, :, -1, :]
                    else:
                        x_val = phys_data[:, :, :]
                else:
                    if self.time_series_input:
                        x_val = obs_data[:, :, -1, :]
                    else:
                        x_val = obs_data[:, :, :]

                MSE_pred = F.mse_loss(h_output, x_val)
                MSE_filtered = F.mse_loss(
                    h_output_filtered[:, :n_step_test, :], x_val[:, :n_step_test, :]
                )
                print(f"{MSE_pred=}")
                print(f"{MSE_filtered=}")
                if self.sigma_block_diag: # self.F.shape == z_dim//2, 2, 2
                    Kmatrix = compute_K_sigma_block_diag(
                        self.lambdas, steps=2, num_real=0, num_complex=self.z_dim // 2
                    )
                else:
                    Kmatrix = compute_K(
                        self.lambdas, steps=2, num_real=0, num_complex=self.z_dim // 2
                    )[1]

                torch.save(Kmatrix, os.path.join(self.save_folder, f"{savename}Kmatrix"))
                torch.save(mu_t_p_list_all, os.path.join(self.save_folder, f"{savename}mu_t_p_list_all"))
                torch.save(mu_t_list_all, os.path.join(self.save_folder, f"{savename}mu_t_list_all"))
                torch.save(sigma_t_p_list_all, os.path.join(self.save_folder, f"{savename}sigma_t_p_list_all"))
                torch.save(sigma_t_list_all, os.path.join(self.save_folder, f"{savename}sigma_t_list_all"))
                torch.save(h_output, os.path.join(self.save_folder, f"{savename}h_output_pred"))
                torch.save(
                    h_output_filtered, os.path.join(self.save_folder, f"{savename}h_output_filtered")
                )
                torch.save(
                    x_val[:, :n_step_test, :], os.path.join(self.save_folder, "target")
                )
                torch.save(
                    obs_data[:, :, :, :], os.path.join(self.save_folder, "obsdata")
                )
                torch.save(self.f_network, os.path.join(self.save_folder, f"{savename}f_network"))
                torch.save(self.G_network, os.path.join(self.save_folder, f"{savename}G_network"))
                torch.save(self.h_network, os.path.join(self.save_folder, f"{savename}h_network"))
                
                return loss, loss_integral, loss_KL, loss_aux

    '''
