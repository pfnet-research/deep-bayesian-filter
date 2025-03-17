import argparse
import torch.multiprocessing as mp
import numpy as np
import os
import tqdm
import torch
from data_assimilation.datasets.doublependulum import DoublePendulum
from data_assimilation.models.dbf_sigma_diag_full import DBF
from data_assimilation.models.dbf_sigma_diag_full_DDP import DBF_DDP, compute_KL_gaussians, compute_loss_integral_VonMises
from read_config import read_config
from torch.utils.data import DataLoader
from torch.distributed import init_process_group, destroy_process_group
#from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ExponentialLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from Lorenz96 import prepare_data, prepare_data_train_test, prepare_data_timeseries, prepare_data_timeseries_train_test

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


#if __name__ == "__main__":
    #mp.set_start_method("spawn")
def main(rank: int, world_size: int):
    ddp_setup(rank, world_size)
    parser = argparse.ArgumentParser(
        description="Deep Bayesian Filter for double pendulum problem"
    )

    parser.add_argument("--config", type=str, help="config file path", required=True)
    args = parser.parse_args()
    config_file = args.config
    config = read_config(f"{config_file}")

    z_dim = int(config["model"]["z_dim"])
    log_sysnoise = float(config["model"]["log_sysnoise"])
    log_obsnoise_model = float(config["model"]["log_obsnoise_model"])
    log_concentration_periodic = float(config["model"]["log_concentration_periodic"])
    periodic_indices = [0, 2]
    nonperiodic_indices = [1, 3]
    block_step = int(config["model"]["block_step"])
    aux_alpha = float(config["model"]["aux_alpha"])
    lr = float(config["model"]["lr"])
    G_val = float(config["model"]["G_val"])
    batch_size = int(config["model"]["batch_size"])
    take_physical_loss = config.getboolean("model", "take_physical_loss")

    dt = float(config["data"]["dt"])
    N_data = int(config["data"]["N_data"])
    obsnoise = float(config["data"]["obsnoise"])
    train_data_seed = int(config["data"]["train_data_seed"])
    test_data_seed = int(config["data"]["test_data_seed"])
    n_step = int(config["data"]["n_step"])
    m_step = int(config["data"]["m_step"])
    n_step_test = int(config["data"]["n_step_test"])
    m_step_test = int(config["data"]["m_step_test"])

    #load_model = config.getboolean("model", "load_model")
    #load_model_path = config["model"]["load_model_path"]
    #jump_step_loss_setting = config["model"]["jump_step_loss_setting"]
    #variable_kernelsize = config.getboolean("model", "variable_kernelsize")
    #kernel = int(config["model"]["kernel"])
    lr_10percent_iters = float(config["model"]["lr_10percent_iters"])
    gamma_per_iter = pow(10, -1.0/lr_10percent_iters)
    #assert jump_step_loss_setting in ["pattern1", "pattern2"], jump_step_loss_setting
    print(f"{take_physical_loss=}")

    outdir = config["others"]["outdir"]

    N_data_test = 100
    load_data = False
    steps_to_generate = n_step + m_step
    take_physical_loss = True
    x_dim = 4

    print("preparing data...")
    #print(f"{data=}")

    train_val = ["fhKGR", "fhKGR"]
    jump_step_loss_setting = "pattern1"

    jump_step_list = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
    if jump_step_loss_setting in ["pattern1"]:
        jump_step_loss_list = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
    elif jump_step_loss_setting in ["pattern2"]:
        jump_step_loss_list = [[2, 3, 4], [1, 1, 1]]
    else:
        print("unknown pattern for jump_step_loss_setting")    
        
    # prepare dataloaders.
    dataset = DoublePendulum(
        num_data=N_data,
        dt=dt,
        n_steps=steps_to_generate,
        obs_noise=obsnoise,
        seed=train_data_seed,
        take_loss_physical=take_physical_loss,
    )
    testset = DoublePendulum(
        num_data=N_data_test,
        dt=dt,
        n_steps=steps_to_generate,
        obs_noise=obsnoise,
        seed=test_data_seed,
        take_loss_physical=take_physical_loss,
    )
    
    #dataset0 = Lorenz96TimeSeries(
    #    num_data=N_data,
    #    dt=dt,
    #    n_steps=steps_to_generate,
    #    time_series_length=time_sequence_length,
    #    obs_noise=obsnoise,
    #    n_grids=N_grids,
    #    obs_data_complete=data,
    #    thinout=False,
    #    take_loss_physical=take_physical_loss,
    #    seed=train_data_seed+rank,
    #    device=rank,
    #)
    #testset = Lorenz96TimeSeries(
    #    num_data=N_data_test,
    #    dt=dt,
    #    n_steps=steps_to_generate,
    #    time_series_length=time_sequence_length,
    #    obs_noise=obsnoise,
    #    n_grids=N_grids,
    #    obs_data_complete=data,
    #    thinout=False,
    #    take_loss_physical=take_physical_loss,
    #    seed=test_data_seed,
    #)
    print(f"{test_data_seed=}")
    #dataloader0 = DataLoader(dataset0, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)#sampler=DistributedSampler(dataset0, num_replicas=world_size, rank=rank))
    #dataloader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=True, num_workers=4)
    #dataloader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=True, num_workers=4)
    #dataloader3 = DataLoader(dataset3, batch_size=batch_size, shuffle=True, num_workers=4)
    #dataloader4 = DataLoader(dataset4, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloader_list = [dataloader]#, dataloader1, dataloader2, dataloader3, dataloader4]
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    print("dataloader ok")
    print(f"{torch.distributed.get_rank()=}")
    print(f"{torch.distributed.get_world_size()=}")

    # prepare models.
    take_blockstep = False
    mode = "doublependulum"
    model = DBF_DDP(
        mode=mode,
        F=None,
        h_network=None,
        take_loss_physical=take_physical_loss,
        take_blockstep=take_blockstep,
        log_sysnoise=log_sysnoise,
        log_obsnoise=log_obsnoise_model,
        log_concentration_periodic=log_concentration_periodic,
        z_dim=z_dim,
        x_dim=x_dim,
        G_val=G_val,
        time_series_input=False,
        time_series_length=1,
        aux_alpha=aux_alpha,
        save_folder=outdir,
        load_model=False,
        load_model_path="",
        variable_kernelsize=False,
        unitmatrix=False,
        G_nondiag=False,
    )
    time_series_input = False
    model.log_obsnoise.requires_grad = True
    model.log_concentration_periodic.requires_grad = True
    parameters = [
        {"params": model.f_network.parameters()},
        {"params": model.G_network.parameters()},
        {"params": model.h_network.parameters()},
        {"params": model.lambdas},
        {"params": model.log_obsnoise},
        {"params": model.log_concentration_periodic},
    ]
    model = DDP(model, device_ids=[rank])
    test_every = min(len(dataloader), 1000)
    #gamma_per_iter = pow(10, -1.0/100000)

    # training loop.
    
    # In this experiment, training data (double pendulum) is dynamically generated.
    # Thus num_epoch makes no sense and fixed to 1.
    for epoch in range(1):
        trainloss_logger = []
        trainloss_integral_logger = []
        trainloss_KL_logger = []
        testloss_logger = []
        testloss_integral_logger = []
        testloss_KL_logger = []
        if epoch == 0:
            #model.module.log_obsnoise.requires_grad = True
            optimizer = torch.optim.Adam(parameters, lr=lr)
            scheduler = ExponentialLR(optimizer, gamma=gamma_per_iter)
            savename = os.path.join(outdir, "fhKGR")
        for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            if ((batch_idx + 1) % test_every == 0) and (rank==0): # test
                model.eval()
                for batch_idx_test, batch in enumerate(testloader):
                    if take_physical_loss:
                        obs_data = batch[0]
                        if time_series_input:
                            target = batch[1][:, :, -1, :]
                        else:
                            target = batch[1][:, :, :]
                    else:
                        obs_data = batch[0]
                        phys_data = None

                    (
                        mu_t_p_list_all,
                        sigma_t_p_list_all,
                        mu_t_list_all,
                        sigma_t_list_all,
                        mu_t_predict,
                        sigma_t_predict,
                        h_output,
                        h_output_filtered,
                        h_results,
                        log_obsnoise,
                        log_concentration_periodic,
                    ) = model(obs_data, n_step, m_step, block_step=1, jump_step=1, return_concentration=True)
                    jump_step_loss = 1
                    loss_KL = compute_KL_gaussians(
                        mu_1=mu_t_list_all[:, ::jump_step_loss, :],
                        sigma_1=sigma_t_list_all[:, ::jump_step_loss, :, :],
                        mu_2=mu_t_p_list_all[:, ::jump_step_loss, :],
                        sigma_2=sigma_t_p_list_all[:, ::jump_step_loss, :, :],
                        z_dim=z_dim,
                        N_data=len(obs_data),
                        n_step=np.ceil((n_step + m_step)/jump_step_loss),
                        sigma_block_diag=True,
                    )
                    loss_integral = compute_loss_integral_VonMises(
                        h_results=h_results,
                        x_value=target,
                        sigma_err=torch.exp(log_obsnoise),
                        periodic_indices=periodic_indices,
                        nonperiodic_indices=nonperiodic_indices,
                        concentration_periodic=torch.exp(log_concentration_periodic),
                    )
                    loss = loss_KL - loss_integral
                    # print(f"{loss=}, {loss_KL=}, {loss_integral=}")
                    #loss_settings = {"jump_step": 1, "jump_step_loss": 1}
                    #loss, loss_integral, loss_KL, pred_simple2 = compute_loss(
                    #    obs_data, phys_data, n_step, m_step, block_step, loss_settings, noKL=False,
                    #)
                    #loss, loss_integral, loss_KL, pred_simple2 = model.module.compute_loss(
                    #obs_data, phys_data, n_step, m_step, block_step, jump_step=1, jump_step_loss=1, noKL=False,
                    #)

                    torch.save(mu_t_p_list_all, f"{savename}_iter{batch_idx+1}_mu_t_p_list_all")
                    torch.save(mu_t_list_all, f"{savename}_iter{batch_idx+1}_mu_t_list_all")
                    torch.save(mu_t_predict, f"{savename}_iter{batch_idx+1}_mu_t_predict")
                    torch.save(sigma_t_p_list_all, f"{savename}_iter{batch_idx+1}_sigma_t_p_list_all")
                    torch.save(sigma_t_list_all, f"{savename}_iter{batch_idx+1}_sigma_t_list_all")
                    torch.save(sigma_t_predict, f"{savename}_iter{batch_idx+1}_sigma_t_predict")
                    torch.save(
                        obs_data[:, :n_step_test+m_step_test, :], f"{savename}_iter{batch_idx+1}_obsdata"
                    )
                    torch.save(
                        target[:, :n_step_test+m_step_test, :], f"{savename}_iter{batch_idx+1}_target"
                    )
                    torch.save(model.module.h_network, f"{savename}_iter{batch_idx+1}_h_network")
                    testloss_logger.append(loss.detach().cpu().numpy())
                    testloss_integral_logger.append(loss_integral.detach().cpu().numpy())
                    testloss_KL_logger.append(loss_KL.detach().cpu().numpy())
                    model.train() # get back to train mode

            if take_physical_loss:
                obs_data = batch[0]
                if time_series_input:
                    target = batch[1][:, :, -1, :]
                else:
                    target = batch[1][:, :, :]
            else:
                obs_data = batch[0]
                phys_data = None
                    
            optimizer.zero_grad()
            #print(f"{obs_data.device=}")
            (
                mu_t_p_list_all,
                sigma_t_p_list_all,
                mu_t_list_all,
                sigma_t_list_all,
                mu_t_predict,
                sigma_t_predict,
                h_output,
                h_output_filtered,
                h_results,
                log_obsnoise,
                log_concentration_periodic,
            ) = model(obs_data=obs_data, n_step=n_step, m_step=m_step, block_step=1, jump_step=1, return_concentration=True)
            jump_step_loss = 1
            loss_KL = compute_KL_gaussians(
                mu_1=mu_t_list_all[:, ::jump_step_loss, :],
                sigma_1=sigma_t_list_all[:, ::jump_step_loss, :, :],
                mu_2=mu_t_p_list_all[:, ::jump_step_loss, :],
                sigma_2=sigma_t_p_list_all[:, ::jump_step_loss, :, :],
                z_dim=z_dim,
                N_data=len(obs_data),
                n_step=np.ceil((n_step + m_step)/jump_step_loss),
                sigma_block_diag=True,
            )
            loss_integral = compute_loss_integral_VonMises(
                h_results=h_results,
                x_value=target,
                sigma_err=torch.exp(log_obsnoise),
                periodic_indices=periodic_indices,
                nonperiodic_indices=nonperiodic_indices,
                concentration_periodic=torch.exp(log_concentration_periodic),
            )
            loss = loss_KL - loss_integral
            #print(f"{loss=}")
            if batch_idx % 20 == 0:
                with open(os.path.join(outdir, "trainloss_integral.txt"), "a") as f:
                    f.write(f"{-loss_integral.detach().cpu().numpy()}\n")
                with open(os.path.join(outdir, "trainloss_KL.txt"), "a") as f:
                    f.write(f"{loss_KL.detach().cpu().numpy()}\n")
                with open(os.path.join(outdir, "log_obsnoise.txt"), "a") as f:
                    f.write(f"{model.module.log_obsnoise.detach().cpu().numpy()}\n")
                with open(os.path.join(outdir, "log_concentration_periodic.txt"), "a") as f:
                    f.write(f"{model.module.log_concentration_periodic.detach().cpu().numpy()}\n")
                with open(os.path.join(outdir, "log_sysnoise.txt"), "a") as f:
                    f.write(f"{model.module.log_sysnoise.detach().cpu().numpy()}\n")
                with open(os.path.join(outdir, "lr.txt"), "a") as f:
                    f.write(f"{scheduler.get_last_lr()}\n")
                # print(f"{loss=}, {loss_integral=}, {loss_KL=}")
                # print(f"{model.module.log_obsnoise=}")
                # print(f"{model.module.log_concentration_periodic=}")
                # print(f"{model.module.log_sysnoise=}")
                
            #print(f"{model.module.log_obsnoise.grad=}")
            loss.backward()
            #print(f"{model.module.log_obsnoise.grad=}")
            #print(f"{model.module.lambdas.grad=}")
            #print(f"{model.module.f_network.fc1.weight.grad=}")
            optimizer.step()
            scheduler.step()
            #print(f"{scheduler.get_last_lr()=}")
            # print(f"{model.module.log_obsnoise.detach().cpu().numpy()=}")
            # print(f"{model.module.log_concentration_periodic.detach().cpu().numpy()=}")

    destroy_process_group()

if __name__ == "__main__":
    world_size = 1  # fixed to 1. DDP is not supported.
    print(f"{world_size=}")
    mp.spawn(main, args=(world_size,), nprocs=world_size)

######################
"""

    model.test(
        testloader,
        n_step_test,
        m_step_test,
        block_step,
        jump_step=1,
        jump_step_loss=1,
        savename=f"beginning_"
    )
    lr_factor_list = [[1, 0.9, 0.8, 0.7, 0.6], [0.5, 0.5/np.sqrt(10), 0.05, 0.05/np.sqrt(10), 0.005]]
    gamma_list = [[1, 1, 1, 1, 1],[pow(10, -1.0/16000) for i in range(5)]]
    # training loop.
    for i_trainval in range(2):
        for i_jumpstep in range(5):
            print(f"{i_trainval=},{i_jumpstep=}")
            model.train(
                num_epochs=1,
                dataloader=dataloader_list[i_jumpstep],
                testloader=testloader,
                train_var=train_val[i_trainval],
                lr=lr*lr_factor_list[i_trainval][i_jumpstep],
                n_step=n_step,
                m_step=m_step,
                n_step_test=n_step_test,
                m_step_test=m_step_test,
                block_step=block_step,
                jump_step=jump_step_list[i_trainval][i_jumpstep],
                jump_step_loss=jump_step_loss_list[i_trainval][i_jumpstep],
                gamma_per_iter=gamma_list[i_trainval][i_jumpstep]
            )
            '''
            model.test(
                testloader,
                n_step_test,
                m_step_test,
                block_step,
                jump_step=jump_step_list[i_trainval][i_jumpstep],
                jump_step_loss=jump_step_loss_list[i_trainval][i_jumpstep],
                savename=f"{train_val[i_trainval]}_jumpstep{jump_step_list[i_trainval][i_jumpstep]}_final_"
            )
            '''
            model.test(
                testloader,
                n_step_test,
                m_step_test,
                block_step,
                jump_step=1,
                jump_step_loss=1,
                savename=f"{train_val[i_trainval]}_jumpstep{jump_step_list[i_trainval][i_jumpstep]}_final_nojumptest_"
            )
        
"""
