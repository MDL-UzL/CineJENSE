import torch
import torch.nn as nn

class TVLoss(nn.Module):
    '''TVLoss'''
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x_complex):
           
        Nx = x_complex.shape[0]
        Ny = x_complex.shape[1]

        tv_loss = 0
        for x in [x_complex.real, x_complex.imag]:
            if x.ndim > 3: #2D total variation; using mean over slices and frames
                # tv_loss += ((torch.sum(torch.abs(x[1:, :, :, :] - x[:Nx - 1, :, :, :]), dim=[0,1,3]) + torch.sum(torch.abs(x[:, 1:, :, :] - x[:, :Ny - 1, :, :]), dim=[0,1,3])) / ((Nx - 1) * (Ny - 1))).mean()
                tv_loss += ((torch.sum(torch.abs(x[1:, :, ...] - x[:Nx - 1, :, ...]), dim=[0,1,-1]) + torch.sum(torch.abs(x[:, 1:, ...] - x[:, :Ny - 1, ...]), dim=[0,1,-1])) / ((Nx - 1) * (Ny - 1))).mean()
               
            else:
                tv_loss += (torch.sum(torch.abs(x[1:, :, :] - x[:Nx - 1, :, :])) + torch.sum(torch.abs(x[:, 1:, :] - x[:, :Ny - 1, :]))) / ((Nx - 1) * (Ny - 1))
        
        return tv_loss
# class LowRankRegularizer(nn.Module):
#     def __init__(self):
#         super(LowRankRegularizer, self).__init__()
#     def forward(self, x_complex):
        
#         _, s, _ = torch.linalg.svd(x_complex, full_matrices=False)
#         norm = s.abs().sum()
        
#         return norm

# class TemporalFFT(nn.Module):
#     ''''TemporalFFT'''
#     def __init__(self):
#         super(TemporalFFT, self).__init__()
    
#     def forward(self, x_complex):
#         Nt = x_complex.shape[1]
        
#         tfftloss = (torch.sum(torch.abs(torch.fft.fftshift(torch.fft.fft(x_complex, dim=-1), dim=-1)), dim=-1) / Nt).mean()
        
#         return tfftloss
    
    
# class TemporalTV(nn.Module):
#     '''TemporalTV'''
#     def __init__(self):
#         super(TemporalTV, self).__init__()
    
#     def forward(self, x_complex):
#         Nt = x_complex.shape[1]
        
#         tvloss = 0
#         for x in [x_complex.real, x_complex.imag]:
#             tvloss += (torch.sum(torch.abs(x[:, 1:] - x[:,:Nt-1]), dim=1) / (Nt - 1)).mean()
        
#         return tvloss
    
# class RelativeL2Loss(nn.Module):
#     def __init__(self, sigma=1.0, reg_weight=0):
#         super(RelativeL2Loss, self).__init__()
#         self.epsilon = 1e-5
#         self.sigma = sigma
#         self.reg_weight = reg_weight
#     def forward(self, input, target):
        
        
#         if input.dtype == torch.float:
#             input = torch.view_as_complex(input) 
#         if target.dtype == torch.float:
#             target = torch.view_as_complex(target)

#         target_max_real = target.abs().max()
#         target /= target_max_real
#         input = input / target_max_real
    
#         loss = 0

#         for x, y in zip([input.real, input.imag], [target.real, target.imag]):
#             magnitude = x.clone().detach()**2
#             scaler = magnitude+self.epsilon
#             squared_loss = (x - y)**2
#             loss += (squared_loss / scaler).mean() 
        
#         return loss
    
