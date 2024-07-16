import torch
from PIL import Image
import matplotlib.pyplot as plt
from train import UNet, transform, denoise_image, add_noise
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = UNet().to(device)
model.load_state_dict(torch.load('ddpm.pth', map_location=device))
model.eval()

# Load and preprocess a custom image
def load_custom_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image.to(device)

image_path = r'path/to/your/custom/image'
image = load_custom_image(image_path, transform)

beta = np.linspace(1e-4, 0.02, 1000)
beta = torch.tensor(beta, dtype=torch.float32).to(device)
alpha = 1 - beta
gamma = torch.cumprod(alpha, dim=0)

t = torch.randint(500, 510, (image.shape[0],)).long().to(device)
gamma_t = gamma[t].view(-1, 1, 1, 1) # Randomly adding noise on clean image to check the model performance
image, _ = add_noise(image, gamma_t)

# Denoise the custom image
with torch.no_grad():
    denoised_image = denoise_image(model, image, num_timesteps=510) # Adjust timestep according to the amount of noise that the image is corrupted with

# Convert tensors to numpy arrays for visualization
denoised_image_np = denoised_image.squeeze().cpu().numpy().transpose(1, 2, 0)

# Visualize the results
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

	
