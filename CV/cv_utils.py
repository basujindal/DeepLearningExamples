
import os
import urllib.request
import zipfile

def download_celeba():

    ### Download the dataset
    url ="https://huggingface.co/datasets/student/celebA/resolve/main/Dataset.zip?download=true"
    curr_file_path = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(curr_file_path, "Dataset.zip")
    print("Downloading CelebA dataset...")
    urllib.request.urlretrieve(url, filename)

    # Unzip the dataset to the current directory
    print("Unzipping CelebA dataset...")

    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(curr_file_path)

    # Remove the zip file
    os.remove(filename)

    os.rename(os.path.join(curr_file_path, "Dataset/"), os.path.join(curr_file_path, "datasets/"))

