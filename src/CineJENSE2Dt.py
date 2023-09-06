import torch
import tinycudann as tcnn
import src.losses as losses
import src.utils as utils
import torch
import torch.cuda.amp as amp

@torch.jit.script
def FFT(x):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(0, 1)), dim=(0, 1)), dim=(0, 1))
@torch.jit.script
def IFFT(x):
    return torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(x, dim=(0, 1)), dim=(0, 1)), dim=(0, 1))

@torch.jit.script
def coil_expand(x, csm):
    return x.unsqueeze(-1) * csm

@torch.jit.script
def coil_reduce(x, csm):
    return (x * torch.conj(csm)).sum(-1)

network_config={
    "otype": "FullyFusedMLP",
    "activation": 'ReLU',
    "output_activation": 'None',
    "n_neurons": 64,
    "n_hidden_layers": 2,
}

encoding_config = {
    'otype': 'Grid',
    'type': 'Hash',
    'n_levels': 16,
    'n_features_per_level': 2,
    'log2_hashmap_size': 19,
    'base_resolution': 16,
    'per_level_scale': 2,
    'interpolation': 'Linear' }

encoding_config_csm = {
    'otype': 'Grid',
    'type': 'Hash',
    'n_levels': 4,
    'n_features_per_level': 8,
    'log2_hashmap_size': 19,
    'base_resolution': 2, 
    'per_level_scale': 1.1,
    'interpolation': 'Linear' }


def CINEJENSE_hash_Recon(tstDsKsp, SamMask, DEVICE, dc_weight, reg_weight, MaxIter, LrImg, LrCsm): 
    

    (nRow, nCol, nFrame, nCoil) = tstDsKsp.shape

    coor = utils.build_coordinate_2Dt_train(nRow, nCol, nFrame, DEVICE).float()
    
    tstDsKsp_tensor = torch.tensor(tstDsKsp, dtype=torch.complex64, device=DEVICE)
    SamMask = torch.tensor(SamMask, dtype=torch.int, device=DEVICE)

   
    reg_loss_function = losses.TVLoss()
    dc_loss_function = torch.nn.HuberLoss(delta=1.0)
    
    IMAGE = torch.compile(tcnn.NetworkWithInputEncoding(3, 2, encoding_config, network_config))
    CSM = torch.compile(tcnn.NetworkWithInputEncoding(3, 2*nCoil, encoding_config_csm, network_config))

    optimizer = torch.optim.Adam(
        params=[
            {"name": "img_net", "params":  list(IMAGE.parameters())}, 
            {"name": "sens_net", "params": list(CSM.parameters()), "learning_rate": LrCsm}, 
        ],
        lr=LrImg,
        betas=(0.9, 0.99),
        eps=1e-15,
    )
    
    scaler = amp.GradScaler()
    for ite_i in range(MaxIter):

        IMAGE.train()
        CSM.train()

        with amp.autocast():
            pre_intensity = IMAGE(coor.view(-1, 3)).float().view(nRow, nCol, nFrame, 2)
            pre_intensity = torch.complex(pre_intensity[...,0:1], pre_intensity[...,1:])

            csm = CSM(coor.view(-1, 3)).float().view(nRow, nCol, nFrame, nCoil, 2)
            csm = torch.complex(csm[...,0:1], csm[...,1:]).squeeze(-1)

            csm_norm = torch.sqrt(torch.sum(csm.conj() * csm, -1)).unsqueeze(-1)
    
            csm = csm / (csm_norm + 1e-12)
            
            SamMask_t = SamMask[:,:,None,:].repeat(1,1,nFrame,1) 

            fft_pre_intensity = FFT(pre_intensity * csm)

            reg_loss = reg_loss_function(pre_intensity)

            dc_loss = dc_loss_function(torch.view_as_real(fft_pre_intensity[SamMask_t == 1]).float(), torch.view_as_real(tstDsKsp_tensor[SamMask_t == 1]).float())
            
            loss = dc_weight * dc_loss + reg_weight * reg_loss 

            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
                
            if (ite_i+1) == MaxIter:
                with torch.no_grad():
                    recon_im = IFFT(fft_pre_intensity * (1 - SamMask_t.float()) + tstDsKsp_tensor)
                    recon_img_reduce = coil_reduce(recon_im,  csm).cpu().detach().numpy()

    return recon_img_reduce

