import os
import yaml
from model import UNet
from scheduler import NoiseScheduler
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import torchvision
from torchvision.utils import make_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Your main code here
def main():
    # Your existing code
    pass

if __name__ == "__main__":
    main()


if __name__ == '__main__':
    # load config
    with open('config.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print("YAML FILE EXCEPTION\n", exc)

    # load model
    model = UNet(config['model']).to(device)
    model.load_state_dict(torch.load(os.path.join(config['train']['checkpoint_folder'], config['train']['checkpoint_name']), map_location=device))
    model.eval()

    # create scheduler
    scheduler = NoiseScheduler(
            algo=config['diffusion']['algo'],
            n_steps=config['diffusion']['n_steps'],
            beta_start=config['diffusion']['beta_start'],
            beta_end=config['diffusion']['beta_end'])

    # random nth sample
    xt = torch.randn(config['sample']['n_samples'],
                     config['model']['img_channels'],
                     config['model']['img_size'],
                     config['model']['img_size']).to(device)

    with torch.no_grad():
        for i in tqdm(reversed(range(config['diffusion']['n_steps'])), total=config['diffusion']['n_steps']):
            # Get prediction of noise
            noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))

            # Use scheduler to get x0 and xt-1
            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

            # Save x0
            imgs = torch.clamp(xt, -1., 1.).detach().cpu()
            imgs = (imgs + 1) / 2
            grid = make_grid(imgs, nrow=int(np.sqrt(config['sample']['n_samples'])))
            img = torchvision.transforms.ToPILImage()(grid)
            if not os.path.exists(os.path.join(config['train']['checkpoint_folder'], 'samples')):
                os.mkdir(os.path.join(config['train']['checkpoint_folder'], 'samples'))
            img.save(os.path.join(config['train']['checkpoint_folder'], 'samples', 'x0_{}.png'.format(i)))
            img.close()
