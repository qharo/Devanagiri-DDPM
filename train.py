import yaml
from dataset import DevanagiriDataset
from torch.utils.data import DataLoader
from model import UNet
from scheduler import NoiseScheduler
import numpy as np
from torch.optim import Adam
import torch
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_parameter_count(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        param = parameter.numel()
        total_params += param
    print(f"Total Trainable Params: {total_params:,}")

if __name__ == '__main__':
    # load config
    with open('config.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print("YAML FILE EXCEPTION\n", exc)

    # create dataset
    deva_data = DevanagiriDataset(config['dataset'])
    deva_loader = DataLoader(deva_data, batch_size=config['train']['batch_size'], shuffle=True, num_workers=4)

    # load model
    model = UNet(config['model']).to(device)
    model.train()

    # create scheduler
    scheduler = NoiseScheduler(
            algo=config['diffusion']['algo'],
            n_steps=config['diffusion']['n_steps'],
            beta_start=config['diffusion']['beta_start'],
            beta_end=config['diffusion']['beta_end'])

    # training params
    num_epochs = config['train']['num_epochs']
    optimizer = Adam(model.parameters(), lr=config['train']['lr'])
    criterion = torch.nn.MSELoss()

    for epoch_id in range(num_epochs):
        losses = []

        for img in tqdm(deva_loader):
            optimizer.zero_grad()
            img = img.float().to(device)

            # Sample random noise
            noise = torch.randn_like(img).to(device)

            # Sample timestep
            t = torch.randint(0, config['diffusion']['n_steps'],  (img.shape[0],)).to(device)

            # Add noise to images according to timestep
            noisy_img = scheduler.add_noise(img, noise, t)
            noise_pred = model(noisy_img, t)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_id + 1,
            np.mean(losses),
        ))
            
