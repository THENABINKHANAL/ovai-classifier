import gradio as gr
import time
from random import random 
import os
from PIL import Image
import json
import pandas as pd


def run(image):
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
