import sys
import re
from pathlib import Path
import argparse

import numpy as np
from numpy import fft
import scipy.io as sio
import torch

if torch.cuda.is_available():
    DEVICE = 'cuda:0'
else:
    DEVICE = 'cpu'
    print("Warning: No GPU found, using CPU instead and tiny-cudann is not available.")

import src.utils as utils
import src.CineJENSE2Dt as CineJENSE2Dt


ID_F = 1000


def run_cinejense(full_mat_path, accelerated_mat_path, mask_mat_path, acceleration, output_dir):
    filename = full_mat_path.name
    save_path = output_dir / filename
    acc_factor = f"{acceleration:02d}"

    try:
        data = utils.loadmat(accelerated_mat_path)[f"kspace_sub{acc_factor}"].transpose(4,3,2,1,0) # nframes, nslices, ncoils, ny, nx -> nx, ny, ncoils,nslices, nframes
    except:
        print(f"Could not load {accelerated_mat_path}, skipping...")
        return

    nx, ny, ncoils, nslices, nframes = data.shape
    try:
        SamMask = utils.loadmat(mask_mat_path)[f"mask{acc_factor}"].transpose(1,0) # ny, nx -> nx, ny
    except:
        print(f"Could not load {mask_mat_path}, creating sampling mask...")
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
            learning_rate=1e-2,
        )

        recon_image[..., _slice, :] = pre_img_dc * NormFactor

    result = {"run4ranking": np.abs(recon_image)}

    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        sio.savemat(
            save_path,
            result,
        )
        print(f"Successfully saved {save_path}.")

    except:
        print(f"Could not save {save_path}.")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./input', help='input directory')
    parser.add_argument('--output', type=str, default='./output', help='output directory')
    parser.add_argument('--dataset', type=str, default='ValidationSet', help='dataset to process: ValidationSet or TestSet')
    parser.add_argument('--tasks', type=str, nargs='+', default='Cine', help='Challenge task: Cine or Mapping')
    parser.add_argument('--coils', type=str, nargs='+', default='Multicoil', help='Singlecoil or Multicoil')
    parser.add_argument('--debug', action='store_true', help='Debug switch')

    args = parser.parse_args()
    input_path = args.input
    base_output_path = args.output
    dataset = args.dataset
    all_tasks = args.tasks
    all_coils = args.coils

    print("Input data store in:", input_path)
    print("Output data store in:", base_output_path)
    all_mats = sorted(list(Path(input_path).glob("**/*.mat")))

    for task in all_tasks:
        for coil in all_coils:
            print(f"Running inference on: {coil} {dataset} {task}")

        if task == "Cine":
            cardiac_views = ['cine_sax','cine_lax']
        elif task == "Mapping":
            cardiac_views = ['T1map','T2map']
        else:
            raise ValueError()

        for vw in cardiac_views:
            # Filter mats for coild and views
            f_mats = all_mats
            f_mats = list(filter(lambda p: coil in str(p), f_mats))
            f_mats = list(filter(lambda p: vw in str(p), f_mats))
            mask_mats = sorted(list(filter(lambda p: str(p).endswith('mask.mat'), f_mats)))
            full_mats = sorted(list(filter(lambda p: "FullSample" in str(p), f_mats)))
            accelerated_mats = list(set(f_mats) - set(mask_mats) - set(full_mats))

            # Create a dictionary based on patient id and acceleration factor. P002, AcFactor04 gets a combined id of 2004, P10, AcFctor10 gets 10010
            mask_mat_dict = {int(m[2])*ID_F + int(m[1]): Path(m[0]) for m in [re.match(r'.*AccFactor([0-9]{2})/P([0-9]{3}).*', str(p)) for p in mask_mats] if m is not None}
            accelerated_mat_dict = {int(m[2])*ID_F + int(m[1]): Path(m[0]) for m in [re.match(r'.*AccFactor([0-9]{2})/P([0-9]{3}).*', str(p)) for p in accelerated_mats] if m is not None}
            full_mat_dict = {int(m[1])*ID_F: Path(m[0]) for m in [re.match(r'.*P([0-9]{3}).*', str(p)) for p in full_mats] if m is not None}

            for id, accelerated_mat_path in accelerated_mat_dict.items():
                pid, acceleration = id // ID_F, id % ID_F
                if id in mask_mat_dict:
                    mask_mat_path = mask_mat_dict[id]
                else:
                    continue

                if pid * ID_F in full_mat_dict:
                    full_mat_path = full_mat_dict[pid * ID_F]
                else:
                    continue

                print(f"Processing P{pid:03d}, AccFactor{acceleration:02d}")
                case_dir = f"{Path(input_path).parts[-1]}/{coil}/{task}/{args.dataset}/AccFactor{acceleration:02d}/P{pid:03d}"

                run_cinejense(
                    full_mat_path, accelerated_mat_path, mask_mat_path, acceleration,
                    Path(base_output_path).resolve() / case_dir
                )
                if args.debug:
                    sys.exit(0)