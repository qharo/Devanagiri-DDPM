import yaml
from dataset import DevanagiriDataset
from model import UNet
from orig_model import Unet

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
    
    # dataset = DevanagiriDataset(config['dataset_params'])
    model = UNet(config['model_params'])
    print_parameter_count(model) 
    
    orig_model = Unet(config['model_params'])
    print_parameter_count(orig_model) 
    #print(dataset[0].shape)
