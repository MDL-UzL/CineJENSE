import numpy as np
import torch
import h5py
import re
import os 

def loadmat(filename):
    """
    Load Matlab v7.3 format .mat file using h5py.
    """
    with h5py.File(filename, "r") as f:
        data = {}
        for k, v in f.items():
            if isinstance(v, h5py.Dataset):
                new_value = v[()][:]

                if new_value.dtype == np.float64:
                    data[k] = new_value
                else:
                    data[k] = new_value["real"] + 1j * new_value["imag"]

            elif isinstance(v, h5py.Group):
                data[k] = loadmat_group(v)
    return data

def loadmat_group(group):
    """
    Load a group in Matlab v7.3 format .mat file using h5py.
    """
    data = {}
    for k, v in group.items():
        if isinstance(v, h5py.Dataset):
            new_value = v[()][:]
            if new_value.dtype == np.float64:
                data[k] = new_value
            else:
                data[k] = new_value["real"] + 1j * new_value["imag"]

        elif isinstance(v, h5py.Group):
            data[k] = loadmat_group(v)
    return data


def build_coordinate_2Dt_train(Nx, Ny, Nt, device='cpu'):
    x = torch.linspace(-1, 1, Nx, device=device)
    y = torch.linspace(-1, 1, Ny, device=device)
    t = torch.linspace(-1, 1, Nt, device=device)

    x, y, t = torch.meshgrid(x, y, t, indexing = 'ij')
    xyt = torch.stack([
        x,
        y, 
        t], -1).view(-1, 3)
    xyt = xyt.view(Nx, Ny, Nt, 3)
    return xyt


def extract_accleration_factor(string, word: str = "AccFactor"):
    '''Extracts the acceleration factor from a string.
    Args:
        string (str): String to extract the acceleration factor from.
        word (str): Word to search for in the string.
    Returns:
        acceleration_factor (str): Acceleration factor.
    '''
    
    pattern = r"\b" + re.escape(word) + r"\D*(\d{2})"
    match = re.search(pattern, string)
    if match:
        return match.group(1)
    else:
        return None

def extract_patient_id(filename):
    '''
    Extracts the patient ID from a filename.
    Args:
        filename (str): path to extract the patient ID from.
        Returns:
        patient_id (str): Patient ID.
    '''
    return re.search(r"(?<=P)\d+", filename).group(0)

def create_directory_structure(root_directory, coil, dataset, acceleration, task):
    """Creates a dictionary with the directory structure of the dataset.
    Args:
        root_directory (str): Root path of the dataset.
        coil (str): Coil configuration. One of 'SingleCoil', 'MultiCoil'.
        dataset (str): Dataset. One of 'TrainingSet', 'ValidationSet', 'TestSet'.
        acceleration (str): Acceleration factor. One of 'AccFactor04', 'AccFactor08', 'AccFactor12'.
        task (str): Task. One of 'Cine', 'Mapping'.
    Returns:
        directory_structure (dict): Dictionary with the directory structure of the dataset.
    """

    directory_structure = {}

    for root, directories, files in os.walk(root_directory):
        current_directory = directory_structure
        path = os.path.relpath(root, root_directory).split(
            os.sep
        )  # Get relative path from root_directory

        for directory in path:
            if directory != ".":
                current_directory = current_directory.setdefault(
                    os.path.basename(directory), {}
                )

        for file in files:
            filename = os.path.splitext(file)[0]  # Remove file extension
            current_directory[filename] = os.path.join(root, file)
            
    directory_structure = directory_structure[coil][task][dataset]
    if dataset == "TrainingSet":
        if acceleration != "FullSample":
            directory_structure_1 = directory_structure[acceleration]
            directory_structure_2 = directory_structure["FullSample"]
            directory_structure = {
                acceleration: directory_structure_1,
                "FullSample": directory_structure_2,
            }
        else:
            directory_structure = directory_structure[acceleration]

    else:
        if acceleration != "FullSample":
            try:
                directory_structure = directory_structure[acceleration]
            except KeyError:
                raise KeyError(
                    f"The given dataset does not contain {acceleration} data."
                )
        else:
            try:
                directory_structure = directory_structure[acceleration]
            except KeyError:
                raise KeyError(
                    "The given dataset does not contain fully sampled data."
                )

    return directory_structure


def extract_file_paths(
    root_path, coil_info, dataset, acceleration_factor, task, cardiac_view="cine_sax"
):
    """Extracts the file paths for the given dataset and coil configuration.
    Args:
        root_path (str): Root path of the dataset.
        coil_info (str): Coil configuration. One of 'SingleCoil', 'MultiCoil'.
        dataset (str): Dataset. One of 'TrainingSet', 'ValidationSet'.
        acceleration_factor (str): Acceleration factor. One of 'AccFactor08', 'AccFactor10', 'AccFactor12'.
        cardiac_view (str): Cardiac view. One of 'cine_sax', 'cine_lax', 'T1map', 'T2map'.
    Returns:
        paths (list): List of file paths of the corresponding undersampeled data.
        path_full (list): List of file paths for the corresponding fully sampled data.
        path_mask (list): List of file paths for the corresponding sampling masks.
    """
    directory_dict = create_directory_structure(
        root_path, coil_info, dataset, acceleration_factor, task)
    paths = []
    path_mask = []
    path_full = []

    def traverse_directory(directory_dict, current_path):
        for key, value in directory_dict.items():
            if isinstance(value, dict):
                # It's a nested directory
                traverse_directory(value, current_path + "/" + key)
            else:
                # It's a file
                if cardiac_view in value and "mask" not in value:
                    if "FullSample" not in value:
                        paths.append(value)
                    else:
                        path_full.append(value)

                elif cardiac_view + "_mask" in value:
                    path_mask.append(value)
                else:
                    pass

    traverse_directory(directory_dict, "")
    return [paths, path_mask, path_full]


