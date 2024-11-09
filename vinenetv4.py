# -*- coding: utf-8 -*-

import os, cv2
import numpy as np
import pandas as pd
import random, tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional
import torch.nn.init as init


import cv2
class CustomDataset(Dataset):
    def __init__(self, root_dir, IDs , transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.IDs = IDs

        self.image_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")

        self.images = os.listdir(self.image_dir)
        self.masks = os.listdir(self.mask_dir)

    def __len__(self):
        return len(self.IDs)

#     def __getitem__(self, idx):

#         img_name = os.path.join(self.image_dir,self.IDs[idx]+ ".png")
#         mask_name = os.path.join(self.mask_dir, self.IDs[idx] + "_instanceIds.png")

#         image = Image.open(img_name)
#         mask = Image.open(mask_name)
#         plt.imshow(mask)
#         plt.show()
#         m = (np.array(mask)).astype('uint8')
#         print("printing mask")
#         print(m.shape)
#         print(np.unique(m,return_counts=True))
#         # mask_array = (np.array(mask)/ 255.0)
#         # mask = Image.fromarray(mask_array)

#         if self.transform:
#             image = self.transform(image)
#             mask = self.transform(mask)

#         return image, mask
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.IDs[idx] + ".png")
        mask_name = os.path.join(self.mask_dir, self.IDs[idx] + "_instanceIds.png")

        # Read image using cv2 in grayscale mode
        image = cv2.imread(img_name)
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)

        # Show mask
#         plt.imshow(mask, cmap='gray')
#         plt.show()

#         print("Printing mask:")
#         print(mask.shape)
#         print(np.unique(mask, return_counts=True))

        if self.transform:
            # Assuming self.transform is a torchvision transform
            # Convert image and mask to PIL Image
            image = Image.fromarray(image)
            mask = Image.fromarray(mask)

            # Apply transformations
            image = self.transform(image)
            mask = self.transform(mask)
        mask = np.where(mask > 0, 1, 0)
#         print(np.unique(mask, return_counts=True))
        return image, mask



# Model Architecture
class DoubleConvolution(nn.Module):

    def __init__(self, in_channels: int,out_channels:int):

        super().__init__()

        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()

        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor):

        x = self.first(x)
        x= self.bn1(x)
        x = self.act1(x)

        x = self.second(x)
        x = self.bn2(x)
        return self.act2(x)


class DownSample(nn.Module):

    def __init__(self):

        super().__init__()

        self.pool = nn.MaxPool2d(2)

    def forward(self, x:torch.Tensor):
        return self.pool(x)

class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):

        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2, stride=2)

    def forward(self, x:torch.Tensor):

        return self.up(x)

class CropAndConcat(nn.Module):
    def forward(self, x:torch.Tensor, contracting_x:torch.Tensor):

        contracting_x = torchvision.transforms.functional.center_crop(contracting_x, [x.shape[2], x.shape[3]])
        x = torch.cat([x,contracting_x],dim = 1)
        return x

# U-Net actual
class UNet(nn.Module):

    def __init__(self,in_channels: int,out_channels:int):

        super().__init__()

        self.down_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in
                                     [(in_channels, 64), (64, 128), (128, 256), (256, 512)]])
        self.down_sample = nn.ModuleList([DownSample() for _ in range(4)])
        self.middle_conv = DoubleConvolution(512, 1024)
        self.up_sample = nn.ModuleList([UpSample(i, o) for i, o in
                                         [(1024, 512), (512, 256), (256, 128), (128, 64)]])
        self.up_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in
                                          [(1024, 512), (512, 256), (256, 128), (128, 64)]])
        self.concat = nn.ModuleList([CropAndConcat() for _ in range(4)])
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        # Initialize the weights using Xavier initialization

        self.initialize_weights()


    def initialize_weights(self):

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)

    def forward(self,x: torch.Tensor):
        pass_through = [] #To collect the outputs of contracting path for later concatenation with the expansive path.

        #Contracting path
        for i in range(len(self.down_conv)):

            x = self.down_conv[i](x) #Two 3×3 convolutional layers
            pass_through.append(x)  # Collect the output

            x = self.down_sample[i](x) #Down-sample
        #Two 3×3 convolutional layers at the bottom of the U-Net
        x = self.middle_conv(x)


        #Expanding path
        for i in range(len(self.up_conv)):
        #Up-sample
            x = self.up_sample[i](x)
            #Concatenate the output of the contracting path
            x = self.concat[i](x,pass_through.pop())
            # Two 3×3 convolutional layers
            x = self.up_conv[i](x)

        #Final 1×1 convolution layer
        x = self.final_conv(x)

        return x

# Dice loss implementation
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        prediction = prediction.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (prediction * target).sum()
        dice_coefficient = (2. * intersection + self.smooth) / (prediction.sum() + target.sum() + self.smooth)

        return 1.0 - dice_coefficient

# Function to save model checkpoint
def save_checkpoint(epoch, model, optimizer, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")

# Define IoU calculation function

def calculate_iou(pred, target):
    intersection = torch.logical_and(pred, target).float().sum((2,3))  # Compute intersection
    union = torch.logical_or(pred, target).float().sum((2,3))  # Compute union
    iou = (intersection + 1e-6) / (union + 1e-6)  # Add a small epsilon to avoid division by zero
    return iou.mean().item()  # Compute mean IoU over the batch



def train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs,checkpoint_interval=5):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_iou = 0.0
        # reg_lambda = 0.001

        # with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:

        for inputs, masks in train_loader:
            inputs, masks = inputs.to(device), masks.to(device).float()

            outputs = model(inputs) #order of dimensions in pytorch is different
            outputs = m(outputs)
            loss = loss_fn(outputs, masks)
                #print(outputs.shape)
                 #print(masks.shape)
                # Add L2 regularization term to the loss
            #l2_reg = torch.tensor(0., device=device)
            #for param in model.parameters():
               # l2_reg += torch.norm(param)
            #loss += reg_lambda * l2_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    

            running_loss += loss.item() * inputs.size(0)
                #  print(np.unique(outputs.cpu().detach().numpy().astype(np.uint8),return_counts=True))
            # Calculate IoU
            outputs = (outputs > 0.5).float()  # Binarize predicted masks
                #  print(np.unique(outputs.cpu().detach().numpy().astype(np.uint8),return_counts=True))
            iou = calculate_iou(outputs, masks)
            running_iou += iou * inputs.size(0)

            # Print the progress bar for the current epoch
            # pbar.update(1)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_iou = running_iou / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for inputs, masks in val_loader:
                inputs, masks = inputs.to(device), masks.to(device).float()
                outputs = m(model(inputs))
                loss = loss_fn(outputs, masks)
                val_loss += loss.item() * inputs.size(0)

                # Calculate IoU
                # outputs = m(outputs) #changes
                outputs = (outputs > 0.5).float()  # Binarize predicted masks
                iou = calculate_iou(outputs, masks)
                val_iou += iou * inputs.size(0)

            val_loss /= len(val_loader.dataset)
            val_iou /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train IoU: {epoch_iou:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_filename = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            save_checkpoint(epoch+1, model, optimizer, checkpoint_filename)

# Train the model
if __name__ == "__main__":
    # # Splitting the dataset into training and testing

    image_folder = "./content/VineNet/images"
    mask_folder = "./content/VineNet/masks"


    # Create a list of image filenames in 'data_path'
    imgs_list = [filename for filename in os.listdir(image_folder)]
    masks_list = [filename for filename in os.listdir(mask_folder)]

    print("images",len(imgs_list))
    print("masks",len(masks_list))

    img_ids = [img[0:-4] for img in imgs_list]

    mask_ids = [m[0:-16]for m in masks_list]
    ids = img_ids

    # Set the seed for reproducibility
    random.seed(55)

    # Shuffle the list of image IDs
    random.shuffle(ids)

    # Split dataset into training and test sets

    # Define the proportions for train, eval, and test sets
    train_ratio = 0.85  # 70% for training
    eval_ratio = 0.15  # 15% for evaluation
    # test_ratio = 0.15  # 15% for testing

    # Calculate the number of images for each set
    num_images = len(ids)
    num_train = int(train_ratio * num_images)
    num_eval = int(eval_ratio * num_images)
    # num_test = int(test_ratio * num_images)

    # Split the image IDs into train, eval, and test sets
    train_ids = ids[:num_train]
    eval_ids = ids[num_train:]
    # test_ids = ids[num_train + num_eval:]

    # Optionally, you can print the lengths of the sets
    print("Number of images in train set:", len(train_ids))
    print("Number of images in eval set:", len(eval_ids))
    # print("Number of images in test set:", len(test_ids))

    # Define transformations for data augmentationm]==
    transform = transforms.Compose([
        transforms.Resize((960, 1920)),
        transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     transforms.ConvertImageDtype(torch.float)
    ])

    ROOT_DIR = "./content/VineNet"
    BATCH = 1
    # Initialize custom dataset
    # dataset = CustomDataset(ROOT_DIR, transform=transform)

    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataset = CustomDataset(ROOT_DIR,train_ids , transform=transform)
    # test_dataset = CustomDataset(ROOT_DIR, test_ids,transform=transform)
    val_dataset = CustomDataset(ROOT_DIR, eval_ids,transform=transform)

    # Create DataLoader for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True,num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False, num_workers=1)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False, num_workers=1)

    for img,m in train_loader:
        print(img.size() )
        print(m.size() )
        break
    # Step 3: Training
    # Get cpu, gpu or mps device for training.

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


    print(f"Using {device} device")

    model = UNet(3,1).to(device)
    # print(model)

    # Optimizing the Model Parameters
    loss_fn = DiceLoss()  # Binary cross-entropy loss
    m = nn.Sigmoid()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # Directory to save checkpoints
    checkpoint_dir = "/scratch/Checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_model(model, train_loader, val_loader, loss_fn,optimizer, num_epochs=250,checkpoint_interval=25)

