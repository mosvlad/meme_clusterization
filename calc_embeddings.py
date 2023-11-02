import os
import pickle
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms

embeddings = {}
filepaths = ["./memes/" + name for name in os.listdir("./memes")]

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
model.eval()

for filepath in tqdm(filepaths):
    try:
        input_image = Image.open(filepath)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(input_batch)

        embeddings[filepath] = output[0]
    except Exception as e:
        print(e)
        pass

f = open("embeddings", "wb")
pickle.dump(embeddings, f)
f.close()
