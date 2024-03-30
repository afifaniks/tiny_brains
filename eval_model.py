import torch
from PIL import Image
from torchvision import transforms

from models.unet import UNet

def load_checkpoint(model, checkpoint: str):
    model.load_state_dict(torch.load(checkpoint))

if __name__ == "__main__":
    checkpoint = "/home/mdafifal.mamun/research/tiny_brains/unet.pth"
    image_path = "/home/mdafifal.mamun/research/tiny_brains/assets/train_images/val/data/CC0002_philips_15_56_M.jpg"

    model = UNet()
    model.eval()

    load_checkpoint(model, checkpoint=checkpoint)

    image = Image.open(image_path).convert('L')
    out = model(transforms.ToTensor()(image).unsqueeze(0))
    out_image = transforms.ToPILImage()(out.squeeze(0))

    out_image.save("out_{}".format(image_path.split("/")[-1]))

