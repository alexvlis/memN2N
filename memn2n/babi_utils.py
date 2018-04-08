# Get Facebook's bAbi dataset
from utils import maybe_download
from shutil import rmtree
import os
import tarfile

def get_babi_en(get_10k=False):
    data_dir = "datasets/tasks_1-20_v1-2/en/"
    if get_10k == True:
        data_dir = "datasets/tasks_1-20_v1-2/en-10k/"
        
    maybe_download('https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz', 'datasets', 11745123)
    file = tarfile.open("datasets/babi_tasks_1-20_v1-2.tar.gz", "r:gz")
    file.extractall("datasets")
    file.close()
    print("Some housekeeping...")
    if not os.path.exists("datasets/babi"):
        os.makedirs("datasets/babi")
    for path, dir, files in os.walk(data_dir):
        for file in files:
            os.rename(os.path.join(data_dir, file), os.path.join("datasets/babi", file))                    
    os.remove("datasets/babi_tasks_1-20_v1-2.tar.gz")
    rmtree("datasets/tasks_1-20_v1-2")
    print("Finished.")