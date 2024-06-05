import gradio as gr
from PIL import Image
import json
import pandas as pd
import torch
import torchvision.transforms as transforms
import timm


def run(image):
    with open('labels.json', 'r') as f:
        label_map = json.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = timm.create_model('resnest50d', pretrained=True, num_classes=len(label_map))
    model.to(device)

    model.load_state_dict(torch.load(f'resnest50d_epoch_200_77_pct.pth', map_location=device))
    model.eval()
    
    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = data_transforms(image)
    images = input_tensor.unsqueeze(0)

    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    #print(outputs)
    #print(predicted)
    
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 10)
    #print(top_prob, top_catid)
    
    results = []
    for i, catid in enumerate(top_catid):
        results.append({'label': list(label_map.keys())[list(label_map.values()).index(int(catid))], 'confidence': float(top_prob[i])})
    

    #results = [{'label': 'species1', 'confidence': 0.9}, {'label': 'species2', 'confidence': 0.8}]
    return pd.DataFrame(results)

gr.Interface(fn=run,
    inputs=[gr.Image(type="pil")],
    outputs=gr.Dataframe(
            headers=["label", "confidence"],
            datatype=["str", "number"],
            col_count=(2, "fixed"),
        ),
    examples=[
        ['test.png'],
    ],
    description="Input an image to get a list of labels").launch(share=True)
