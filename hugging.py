

import skimage
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms

import torchxrayvision as xrv

model_name = "densenet121-res224-mimic_nb"

img_url = "/Users/yashpradhan/Desktop/0a1f5edc85aa58d5780928cb39b08659c1fc4d6d7c7dce2f8db1d63c7c737234_big_gallery.jpeg"

model = xrv.models.get_model(model_name)

img = skimage.io.imread(img_url)
img = xrv.datasets.normalize(img, 255)

# Check that images are 2D arrays
if len(img.shape) > 2:
    img = img[:, :, 0]
if len(img.shape) < 2:
    print("error, dimension lower than 2 for image")

# Add color channel
img = img[None, :, :]

transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop()])

img = transform(img)

with torch.no_grad():
    img = torch.from_numpy(img).unsqueeze(0)
    preds = model(img).cpu()
    output = {
        k: float(v)
        for k, v in zip(xrv.datasets.default_pathologies, preds[0].detach().numpy())
    }
print(output)
