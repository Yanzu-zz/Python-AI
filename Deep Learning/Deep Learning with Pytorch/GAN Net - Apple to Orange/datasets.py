import os
import glob
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as trForms


class ImageDataset(Dataset):
    def __init__(self, rootPath="", transform=None, model="train"):
        self.transform = trForms.Compose(transform)

        # A 里面是苹果
        # B 里面是橘子
        # 我们要将他俩互相转换
        self.pathA = os.path.join(rootPath, model, "A/*")
        self.pathB = os.path.join(rootPath, model, "B/*")
        self.list_A = glob.glob(self.pathA)
        self.list_B = glob.glob(self.pathB)

    def __getitem__(self, index):
        im_pathA = self.list_A[index % len(self.list_A)]
        im_pathB = random.choice(self.list_B)

        im_A = Image.open(im_pathA)
        im_B = Image.open(im_pathB)
        item_A = self.transform(im_A)
        item_B = self.transform(im_B)

        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.list_A), len(self.list_B))


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    root = "./datasets/apple2orange"

    transform_ = [trForms.Resize(256, Image.BILINEAR), trForms.ToTensor()]
    dataloader = DataLoader(ImageDataset(root, transform_, "train"),
                            batch_size=1, shuffle=True, num_workers=1)

    for i, batch in enumerate(dataloader):
        print(i)
        print(batch)
