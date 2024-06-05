import gradio as gr
import time
from random import random 
import os
from PIL import Image
import json
import pandas as pd
import torch
import torchvision.models as models


def run(image):
    with open('./dataset/labels.json', 'r') as f:
        label_map = json.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50()
    model.to(device)

    model.load_state_dict(torch.load(f'resnest50d_epoch_200_77_pct.pth', map_location=device))
    model.eval()
    
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = data_transforms(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        results = model(input_batch)
    print(results)

    results = [{'label': 'species1', 'confidence': 0.9}, {'label': 'species2', 'confidence': 0.8}]
    return pd.DataFrame(results)

gr.Interface(fn=run,
    inputs=[gr.Image(type="pil")],
    outputs=gr.Dataframe(
            headers=["label", "confidence"],
            datatype=["str", "number"],
            col_count=(2, "fixed"),
        ),
    examples=[
        ['test.jpg'],
    ],
    description="Input an image to get a list of labels").launch(share=True)
