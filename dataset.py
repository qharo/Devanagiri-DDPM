import torchvision.transforms as T
import glob
from PIL import Image
from zipfile import ZipFile
from io import BytesIO
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import os
import requests
import subprocess

class DevanagiriDataset(Dataset):
    def __init__(self, params: dict):
        self.params = params

        # for Devanagiri Handwritten Dataset
        download_url = "https://archive.ics.uci.edu/static/public/389/devanagari+handwritten+character+dataset.zip"
        
        self.download_dataset(url=download_url)
        self.imgs = self.load_dataset()

    def load_dataset(self):
        imgs = []

        # find all PNG/file_ext files within folder
        for root, dirs, files in os.walk(self.params['save_path']):
            imgs += [os.path.join(root, file) for file in files if file.endswith(self.params['img_ext'])]

        print(f"Found {len(imgs)} images")
        return imgs

    
    def download_dataset(self, url: str):
            # if data exists, move on
            if os.path.exists(self.params['save_path']):
                print('Data found')
                return 0

            # make dir if doesn't exist
            os.makedirs(self.params['save_path'], exist_ok=True)

            # get streaming response
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

                # extract file
                if url.split('/')[-1].endswith('.zip'):
                    with ZipFile(content) as zip_file:
                        
                        total_files = len(zip_file.infolist())
                        self.total = total_files
                        for file in tqdm(zip_file.infolist(), total=total_files, desc="Extracting"):
                            zip_file.extract(file, self.params['save_path'])
                    print(f"Dataset extracted to {self.params['save_path']}")

            else:
                print(f"Failed to download the dataset. Status code: {response.status_code}")

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
       # within [-1, 1] range 
        transform = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: 2 * x - 1)
        ])

        with Image.open(self.imgs[index]) as im:
            return transform(im)
