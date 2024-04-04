import torch
from PIL import Image
from torchvision import transforms
import os

from tqdm import tqdm

from models.unet import UNet


def load_checkpoint(model, checkpoint: str):
    model.load_state_dict(torch.load(checkpoint))


def predict(model, device, image_path, output_path):
    image = Image.open(image_path).convert('L')
    image = transforms.ToTensor()(image).unsqueeze(0)
    image = image.to(device)
    out = model(image)
    out_image = transforms.ToPILImage()(out.squeeze(0))

    out_image.save(output_path)


if __name__ == "__main__":
    curr_dir = os.getcwd()

    checkpoint = os.path.join(curr_dir, "test_contrast_inverted.pth")
    image_dir = os.path.join(curr_dir, "assets", "contrast_inverted_train_images", "test", "data")
    output_dir = os.path.join(curr_dir, "assets", "model_outputs", "contrast_inverted")

    images = os.listdir(image_dir)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet()
    model = model.to(DEVICE)
    model.eval()

    load_checkpoint(model, checkpoint=checkpoint)

    for image in tqdm(images):
        predict(model, DEVICE, os.path.join(image_dir, image), os.path.join(output_dir, image))
