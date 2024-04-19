import pandas as pd
import numpy as np
from scipy.io import savemat
from scipy.io import loadmat
import os


def list_files_in_directory(directory):
    try:        
        files = os.listdir(directory)
        return files
    except Exception as e:
        print("Error listing files:", e)
        return None
    
import os
import imageio

def create_video(input_directory, output_name='spectrogram.mp4', fps=1):
    """
    Creates a video from a sequence of PNG images in a directory.

    Arguments:
    input_directory (str): Path to the directory where the images are stored.
    output_name (str): Name of the output video file (default is 'output.mp4').
    fps (int): Frames per second of the video (default is 1).

    Returns:
    None
    """
    # List all PNG files in the directory
    image_files = [f for f in os.listdir(input_directory) if f.endswith('.png')]

    # Sort the files in alphabetical order
    image_files.sort()

    # List to store the full paths of image files
    image_paths = [os.path.join(input_directory, f) for f in image_files]

    # List to store read images
    images = []

    # Read each image and append it to the list of images
    for path in image_paths:
        images.append(imageio.imread(path))

    # Output path for the video
    output_path = output_name

    # Save the video
    imageio.mimsave(output_path, images, fps=fps)

    print("Video successfully created at:", output_path)
    

def create_directory_structure(base_dir):
    # Lista de subdiretórios (classes)
    subdirs = ['normal', 'inner_fault', 'outer_fault', 'ball_fault', 'cage_fault']

    # Cria o diretório base se ele não existir
    os.makedirs(base_dir, exist_ok=True)

    # Cria os subdiretórios
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)


def read_csv_data(file_name):
    try:
        # Reading the CSV file
        data = pd.read_csv(file_name, header=None)
        
        # Renaming the columns
        column_names = ['Tachometer', 'Underhang_Axial', 'Underhang_Radial', 'Underhang_Tangential',
                        'Overhang_Axial', 'Overhang_Radial', 'Overhang_Tangential', 'Microphone']
        data.columns = column_names
        
        return data
    except Exception as e:
        print("Error reading the CSV file:", e)
        return None
    

def save_as_numpy_array(data, file_name):
    try:
        # Convert DataFrame to numpy array
        numpy_array = data.to_numpy()
        
        # Save numpy array to file
        np.save(file_name, numpy_array)
        print("Data saved as numpy array successfully.")
    except Exception as e:
        print("Error saving data as numpy array:", e)



def save_as_matlab(data, file_name):
    try:
        # Convert DataFrame to dictionary
        data_dict = data.to_dict(orient='list')
        
        # Save data as MATLAB .mat file
        savemat(file_name, data_dict)
        print("Data saved as MATLAB .mat file successfully.")
    except Exception as e:
        print("Error saving data as MATLAB .mat file:", e)



def read_matlab_file(file_name):
    try:
        # Load MATLAB .mat file
        data = loadmat(file_name)
        return data
    except Exception as e:
        print("Error reading MATLAB .mat file:", e)
        return None
    

def get_file_size(file_name):
    try:
        # Get the size of the file in bytes
        size_bytes = os.path.getsize(file_name)
        return size_bytes
    except Exception as e:
        print("Error getting file size:", e)
        return None
    

def get_subdirectories(parent_directory):
    subdirectories = []
    for dirpath, dirnames, filenames in os.walk(parent_directory):
        if not dirnames:
            subdirectories.append(dirpath)
    return subdirectories    


def total_size_of_files(directory):
    total_size = 0
    try:
        # Iterate over all files in the directory
        for dirpath, _, filenames in os.walk(directory):
            for filename in filenames:
                # Get the full path of the file
                file_path = os.path.join(dirpath, filename)
                # Get the size of the file and add it to the total size
                total_size += os.path.getsize(file_path)
        return total_size
    except Exception as e:
        print("Error calculating total size of files:", e)
        return None
    
def create_directory(directory):
    try:
        # Check if the directory already exists
        if not os.path.exists(directory):
            # Create the directory if it doesn't exist
            os.makedirs(directory)
            print("Directory created successfully:", directory)
        else:
            print("Directory already exists:", directory)
    except Exception as e:
        print("Error creating the directory:", e)