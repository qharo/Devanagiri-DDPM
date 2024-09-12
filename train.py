import yaml
from dataset import DevanagiriDataset

if __name__ == '__main__':
    # load config
    with open('config.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print("YAML FILE EXCEPTION\n", exc)
    
    dataset = DevanagiriDataset(config['dataset_params'])
    print(dataset[0].shape)
