from zipfile import ZipFile
from io import BytesIO
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import os
import requests
import subprocess

class DevanagiriDataset(Dataset):
    def __init__(self, params: dict):
        self.download_dataset(save_path=params['save_path'])

    def download_dataset(self, save_path='data/', url="https://archive.ics.uci.edu/static/public/389/devanagari+handwritten+character+dataset.zip"):
            # make dir if doesn't exist
            os.makedirs(save_path, exist_ok=True)

            # get file
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                filename = url.split('/')[-1]

                total_size = int(response.headers.get('content-length', 0)) 
                progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f'Downloading {filename}')
                
                # save file
                content = BytesIO()
                for data in response.iter_content(chunk_size=1024):
                    size = content.write(data)
                    progress_bar.update(size)
                progress_bar.close()

                if url.split('/')[-1].endswith('.zip'):
                    with ZipFile(content) as zip_file:
                        
                        total_files = len(zip_file.infolist())
                        for file in tqdm(zip_file.infolist(), total=total_files, desc="Extracting"):
                            zip_file.extract(file, save_path)
                    print(f"Dataset extracted to {save_path}")


                command = f"mv {os.path.join(save_path, 'DevanagariHandwrittenCharacterDataset')}/* {save_path}"
                
                try:
                    result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    result = subprocess.run(f"rm -rf {os.path.join(save_path, 'DevanagariHandwrittenCharacterDataset')}/", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) 

                except subprocess.CalledProcessError as e:
                     print(f"An error occurred while moving files: {e}")
                     print(e.stderr)

            else:
                print(f"Failed to download the dataset. Status code: {response.status_code}")

