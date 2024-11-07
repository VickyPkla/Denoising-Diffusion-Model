import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from train import ConditionalUNet, transform, denoise_image_with_condition, add_noise
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os

def calculate_rmse(image1, image2):
    return np.sqrt(np.mean((image1 - image2) ** 2))

def evaluate_performance(denoised_image, ground_truth_image):
    # Ensure both images are in the range [0, 1]
    ground_truth_image = (ground_truth_image - ground_truth_image.min()) / (ground_truth_image.max() - ground_truth_image.min())
    denoised_image = (denoised_image - denoised_image.min()) / (denoised_image.max() - denoised_image.min())
    
    # Ensure images are at least 7x7 pixels for SSIM calculation
    if ground_truth_image.shape[0] < 7 or ground_truth_image.shape[1] < 7:
        raise ValueError("Images are too small for SSIM calculation. Resize your images or handle them differently.")

    # Calculate SSIM
    ssim_value = ssim(ground_truth_image, denoised_image, data_range=1.0, win_size=3, multichannel=True)

    # Calculate RMSE
    rmse_value = calculate_rmse(ground_truth_image, denoised_image)

    # Calculate PSNR
    psnr_value = psnr(ground_truth_image, denoised_image, data_range=1.0)

    return ssim_value, rmse_value, psnr_value


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = ConditionalUNet().to(device)
model.load_state_dict(torch.load('advfringe2.pth', map_location=device), strict=False)
model.eval()

# Load and preprocess a custom LR image
def load_custom_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image.to(device)

image_path = r'Fringe data/clean/Image11Y_306.png'  # Replace with your LR image path
image = load_custom_image(image_path, transform)

beta = np.linspace(1e-4, 0.02, 1000)
beta = torch.tensor(beta, dtype=torch.float32).to(device)
alpha = 1 - beta
gamma = torch.cumprod(alpha, dim=0)

t = torch.randint(10, 100, (image.shape[0],)).long().to(device)
gamma_t = gamma[t].view(-1, 1, 1, 1)
gt = image
image, _ = add_noise(image, gamma_t)

# Denoise the custom LR image to generate HR image
with torch.no_grad():
    denoised_image = denoise_image_with_condition(model, image, num_timesteps=t)

# Convert tensors to numpy arrays for visualization
denoised_image_np = denoised_image.squeeze().cpu().numpy().transpose(1, 2, 0)
gt_np = gt.squeeze().cpu().numpy().transpose(1, 2, 0)

# Evaluate performance
ssim_value, rmse_value, psnr_value = evaluate_performance(denoised_image_np, gt_np)
print(f"SSIM: {ssim_value}, RMSE: {rmse_value}, PSNR: {psnr_value}")



# def load_image(image_path, transform, device):
#     image = Image.open(image_path).convert('RGB')
#     image = transform(image).unsqueeze(0)
#     return image.to(device)

# def evaluate_folder(clean_folder, transform, device):
#     model = ConditionalUNet().to(device)
#     model.load_state_dict(torch.load('advfringe2.pth', map_location=device), strict=False)
#     model.eval()

#     ssim_scores = []
#     rmse_scores = []
#     psnr_scores = []

#     for filename in os.listdir(clean_folder):
#         clean_image_path = os.path.join(clean_folder, filename)

#         clean_image = load_image(clean_image_path, transform, device)

#         t = torch.randint(50, 100, (clean_image.shape[0],)).long().to(device)
#         beta = np.linspace(1e-4, 0.02, 1000)
#         beta = torch.tensor(beta, dtype=torch.float32).to(device)
#         alpha = 1 - beta
#         gamma = torch.cumprod(alpha, dim=0)
#         gamma_t = gamma[t].view(-1, 1, 1, 1)

#         noisy_image, _ = add_noise(clean_image, gamma_t)

#         with torch.no_grad():
#             denoised_image = denoise_image_with_condition(model, noisy_image, num_timesteps=t)

#         denoised_image_np = denoised_image.squeeze().cpu().numpy().transpose(1, 2, 0)
#         clean_image_np = clean_image.squeeze().cpu().numpy().transpose(1, 2, 0)

#         ssim_value, rmse_value, psnr_value = evaluate_performance(denoised_image_np, clean_image_np)

#         ssim_scores.append(ssim_value)
#         rmse_scores.append(rmse_value)
#         psnr_scores.append(psnr_value)

#     avg_ssim = np.mean(ssim_scores)
#     avg_rmse = np.mean(rmse_scores)
#     avg_psnr = np.mean(psnr_scores)

#     return avg_ssim, avg_rmse, avg_psnr

# # Path to the clean image folder
# clean_folder = r'Fringe data/evaluate'

# # Evaluate all images in the folder
# avg_ssim, avg_rmse, avg_psnr = evaluate_folder(clean_folder, transform, device)

# print(f"Average SSIM: {avg_ssim}, Average RMSE: {avg_rmse}, Average PSNR: {avg_psnr}")




# Visualize

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow((image.squeeze().cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5).clip(0, 1))
plt.title('Input Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow((denoised_image_np * 0.5 + 0.5).clip(0, 1))  # Denormalizing the image
plt.title('Output Image')
plt.axis('off')

plt.show()




