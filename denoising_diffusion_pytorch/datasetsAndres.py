import torch
import numpy as np
from torch.utils.data import Dataset
import os
from torchvision import transforms
from PIL import Image
    
    
class LectorCubos(Dataset):
    def __init__(self, resolucion):

        try:
            self. archivos = os.listdir(f'./Dataset/DataOut{resolucion}metro')
            self.cubos = [f'./Dataset/DataOut{resolucion}metro/' + cubo for cubo in self.archivos]
        except:
            pass

        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])

    def __len__(self):
        # return len(self.cubos)
        return 10000

    def __getitem__(self, idx):
        cubo = np.load(self.cubos[idx])
        cubo = torch.from_numpy(cubo).float()

        state = torch.get_rng_state()
        cubo = self.augmentations(cubo)

        return cubo*2 - 1

if __name__ == "__main__":
    pass