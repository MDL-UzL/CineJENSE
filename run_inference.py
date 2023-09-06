import argparse
import os
import sys

import numpy as np
import torch
from numpy import fft

if torch.cuda.is_available():
    DEVICE = 'cuda:0'
else:
    DEVICE = 'cpu'
    
    
import scipy.io as sio
import src.utils as utils
from src.utils import (extract_accleration_factor, extract_file_paths,
                       extract_patient_id)

import src.CineJENSE2Dt as CineJENSE2Dt



def run_cinejense(file_structure, cardiac_view, output_path):
   
    for i in range(len(file_structure[0])):  
        filename = [sublist[i] for sublist in file_structure if sublist]
        if not os.path.exists(filename[0]):
            print(f"File {filename[0]} does not exist")
            return
        
        if not os.path.exists(os.path.join(output_path, filename[0].split("/input/")[-1].split(cardiac_view)[0]) ):
            os.makedirs(
                os.path.join(
                    output_path, filename[0].split("/input/")[-1].split(cardiac_view)[0]
                )
            )
            
        acc_factor = extract_accleration_factor(filename[0])
     
        try:
            data = utils.loadmat(filename[0])[f"kspace_sub{acc_factor}"].transpose(4,3,2,1,0) # nframes, nslices, ncoils, ny, nx -> nx, ny, ncoils,nslices, nframes                
        except:
            print(f"Could not load {filename[0]}, skipping...")
            return
        
        print(f"Processing Patient {extract_patient_id(filename[0])}")
        nx, ny, ncoils, nslices, nframes = data.shape
        try: 
            SamMask = utils.loadmat(filename[1])[f"mask{acc_factor}"].transpose(1,0) # ny, nx -> nx, ny
        except:
            print(f"Could not load {filename[1]}, creating sampling mask...")
            SamMask = (~(np.abs(data[:,:,0,0,0]) == 0))[..., np.newaxis].repeat(ncoils, axis=-1).astype(float)
            
        
        if SamMask.shape[0] != nx or SamMask.shape[1] != ny:
            print("Creating sampling mask due to shape mismatch between k-space and mask...")
            SamMask = (~(np.abs(data[:,:,0,0,0]) == 0))[..., np.newaxis].repeat(ncoils, axis=-1).astype(float)
        else:
            SamMask = SamMask[..., np.newaxis].repeat(ncoils, axis=-1)
        
        recon_image = np.zeros((nx, ny, nslices, nframes), np.complex64)

        for _slice in range(nslices):
            print(f"Slice {_slice+1}/{nslices}")
            tstDsKsp = data[..., _slice, :].transpose(0,1,3,2)  #  nx, ny, nt, nc
        
            zf_coil_img = fft.fftshift(
                fft.ifft2(fft.fftshift(tstDsKsp, axes=(0, 1)), axes=(0, 1)),
                axes=(0, 1),
            )
            zf_coil_combined = np.sqrt(np.sum(np.abs(zf_coil_img) ** 2, axis=-1)) 
            NormFactor = np.max(zf_coil_combined)     

            tstDsKsp = tstDsKsp / NormFactor
    
                    
            pre_img_dc = CineJENSE2Dt.CINEJENSE_hash_Recon(
                tstDsKsp,
                SamMask,
                DEVICE,
                dc_weight = 1.0,
                reg_weight = 5,
                MaxIter = 200,
                LrImg = 1e-2,
                LrCsm = 1e-2,
            )
            
            pre_img_dc = np.zeros((nx, ny, nframes), np.complex64)
            recon_image[..., _slice, :] = pre_img_dc * NormFactor

        result = {"run4ranking": np.abs(recon_image)}

        save_path = os.path.join(
            output_path, filename[0].split("/input/")[-1]
        )
        
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
            
        print(f"Saving to {save_path}...")
        
        try:
            sio.savemat(
                save_path,
                result,
            )
            print(f"Successfully saved Patient {extract_patient_id(filename[0])}!")
            
        except:
            print(f"Could not save Patient {extract_patient_id(filename[0])}, skipping...")
            return
        
        


if __name__ == "__main__":
    argv = sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, nargs='?', default='/input', help='input directory')
    parser.add_argument('--output', type=str, nargs='?', default='/output', help='output directory')
    parser.add_argument('--dataset', type=str, nargs='?', default='ValidationSet', help='Dataset to process: ValidationSet or TestSet')
    parser.add_argument('--task', type=str, nargs='?', default='Cine', help='Challenge task: Cine or Mapping')
    parser.add_argument('--coil', type=str, nargs='?', default='Multicoil', help='Singlecoil or Multicoil')
    
    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    DATASET = args.dataset
    TASK = args.task
    COIL = args.coil
    

    print("Input data store in:", input_path)
    print("Output data store in:", output_path)
    print(f"Running inference on: {COIL} {DATASET} {TASK}")
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    accFactors = ['AccFactor04', 'AccFactor08', 'AccFactor10']
    
    if TASK == "Cine":
        cardiac_views = ['cine_sax','cine_lax']
    else:
        cardiac_views = ['T1map','T2map']
    for coil in [COIL]:
        print('Processing:', coil)
        for accFactor in accFactors: 
            print(accFactor)
            for cardiac_view in cardiac_views: 
                print(cardiac_view)
                try:
                    file_structure = extract_file_paths(input_path, COIL, DATASET, accFactor, TASK, cardiac_view) 
                    run_cinejense(file_structure, cardiac_view, output_path)
                    
                except KeyError:
                    print(f"No Dataset given the keys: {DATASET}, {accFactor}, {TASK}, {cardiac_view}, {COIL}, skipping...")
                    continue
           
        