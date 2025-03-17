import torch

torch.set_default_dtype(torch.float64)
device = "cuda"


def form_tensor(omegas: torch.Tensor, steps: int):
    cosine = torch.cos(omegas)
    sine = torch.sin(omegas)
    output_tensor = torch.empty((steps, 2, 2), device=omegas.device)
    output_tensor[:, 0, 0] = cosine
    output_tensor[:, 0, 1] = -sine
    output_tensor[:, 1, 0] = sine
    output_tensor[:, 1, 1] = cosine
    return output_tensor
'''
def compute_K_sigma_block_diag(lambdas: torch.Tensor, steps, num_real=0, num_complex=1):
    z_dim = num_real + num_complex * 2
    K_matrix = torch.zeros((steps, z_dim//2, 2, 2), device=lambdas.device)
    for i in range(num_complex):
        ind = 2 * i
        mus = 0.01 * torch.arange(steps, device=lambdas.device) * lambdas[ind]
        omegas = torch.arange(steps, device=lambdas.device) * lambdas[ind + 1]
        Jordan_block = torch.exp(mus).view(steps, 1, 1) * form_tensor(omegas, steps)
        K_matrix[:, i] = Jordan_block

    for j in range(num_real):
        ind = 2 * num_complex + j
        mus = lambdas[ind : ind + 1]
        K_matrix[:, i] = torch.exp(mus)

    return K_matrix
'''
def compute_K_sigma_block_diag(lambdas: torch.Tensor, steps, num_real=0, num_complex=1):
    z_dim = num_real + num_complex * 2
    mus = 0.01 * lambdas[0::2]
    omegas = lambdas[1::2]
    cosine = torch.cos(omegas).view(num_complex, 1)
    sine = torch.sin(omegas).view(num_complex, 1)
    Jordan_block = torch.exp(mus).view(num_complex, 1, 1) * torch.cat((cosine, -sine, sine, cosine), dim=1).view(num_complex, 2, 2)
    '''
    K_matrix = torch.zeros((steps, z_dim//2, 2, 2), device=lambdas.device)
    for i in range(num_complex):
        ind = 2 * i
        mus = 0.01 * torch.arange(steps, device=lambdas.device) * lambdas[ind]
        omegas = torch.arange(steps, device=lambdas.device) * lambdas[ind + 1]
        Jordan_block = torch.exp(mus).view(steps, 1, 1) * form_tensor(omegas, steps)
        K_matrix[:, i] = Jordan_block

    for j in range(num_real):
        ind = 2 * num_complex + j
        mus = lambdas[ind : ind + 1]
        K_matrix[:, i] = torch.exp(mus)
    '''
    
    return Jordan_block

def compute_K(lambdas: torch.Tensor, steps, num_real=0, num_complex=1):
    z_dim = num_real + num_complex * 2
    K_matrix = torch.zeros((steps, z_dim, z_dim), device=lambdas.device)
    for i in range(num_complex):
        ind = 2 * i
        mus = 0.01 * torch.arange(steps, device=lambdas.device) * lambdas[ind]
        omegas = torch.arange(steps, device=lambdas.device) * lambdas[ind + 1]
        Jordan_block = torch.exp(mus).view(steps, 1, 1) * form_tensor(omegas, steps)
        K_matrix[:, ind : ind + 2, ind : ind + 2] = Jordan_block

    for j in range(num_real):
        ind = 2 * num_complex + j
        mus = lambdas[ind : ind + 1]
        K_matrix[:, ind, ind] = torch.exp(mus)

    return K_matrix

def form_tensors_for_G(G_output):
    pass
