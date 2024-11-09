import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from vinenetv4 import UNet 
import torch.nn.functional as F

# Function to generate masks for images
def generate_masks(model, input_folder, output_folder, transform):
    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
        print("img_shape:" ,image_tensor.size())
        with torch.no_grad():
            output = model(image_tensor)
            print("out: ",output.size())
            output = sigmoid(output)
            print("out: ",output.shape," type: ", type(output))

            output = (output > 0.5).float()  # Binarize the output mask

        output_image = transforms.ToPILImage()(output.squeeze(0))
        output_image.save(os.path.join(output_folder, f"mask_{filename}"))
    print(f"Saved all the images to {output_folder}")
if __name__ == "__main__":

    # Define transformations for input images
    transform = transforms.Compose([
        transforms.Resize((960, 1920)),
        transforms.ToTensor()
    ])
    sigmoid = torch.nn.Sigmoid()
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    model = UNet(3,1).to(device)
    model.load_state_dict(torch.load("/scratch/Checkpoints/checkpoint_epoch_25.pt",map_location=device)['model_state_dict'])

    # Set the model to evaluation mode
    model.eval()

    # Create output directory for final masks
    input_dir = "/scratch/Test_Final/images"
    output_dir = "/scratch/Test_Final/Masks/"
    os.makedirs(output_dir, exist_ok=True)

    # Generate masks for images in the input folder and save them in the output folder
    generate_masks(model, input_dir,output_dir, transform)

# Commented out IPython magic to ensure Python compatibility.
# from PIL import Image
# import matplotlib.pyplot as plt
# import os
# %matplotlib inline
# output_dir = "./Final_masks_/"
# th to the image file
# dir_path = output_dir
# test_path = "/kaggle/working/content/drive/MyDrive/Dataset/data/Test_final/images"
# fig, axes = plt.subplots(5, 2, figsize=(20,10))
# for i in range(5):
#     names = os.listdir("./Final_masks_/")
#     mask_path= os.path.join(dir_path,names[i])
# #     names2  = os.listdir(test_path)
#     img_path  = os.path.join(test_path,names[i][5:])


#     img = Image.open(img_path)
#     mask = Image.open(mask_path)

#     # Create a figure and two subplots


#     # Plot the first image on the left subplot
#     axes[i,0].imshow(img)
# #     axes[i,0].set_title('original')

#     # Plot the second image on the right subplot
#     axes[i,1].imshow(mask,cmap='gray')
# #     axes[i,1].set_title('Masked')

#     # Hide axis labels
# #     for ax in axes:
# #         ax.axis('off')

#     # Show the images
# plt.show()