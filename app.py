import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import pandas as pd
import torchvision.transforms as transforms

# Define Models
class MSSM_SCDNet(torch.nn.Module):
    def __init__(self, num_classes=7):
        super(MSSM_SCDNet, self).__init__()
        from transformers import SegformerForSemanticSegmentation
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            'nvidia/segformer-b0-finetuned-ade-512-512',
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        self.cam = torch.nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sam = torch.nn.Sequential(
            torch.nn.Conv2d(num_classes, num_classes // 2, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(num_classes // 2, num_classes, kernel_size=3, padding=1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        outputs = self.segformer(x)
        logits = outputs.logits
        logits = torch.nn.functional.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        cam_out = self.cam(logits)
        sam_out = self.sam(logits)
        logits = logits * sam_out
        return logits

# Initialize Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
segmenter = MSSM_SCDNet(num_classes=7).to(device)

# Load Trained Weights
segmenter.load_state_dict(torch.load('/content/drive/MyDrive/WildfireProject/models/segmenter_retrained.pth'))
segmenter.eval()

# Streamlit App
st.title("Wildfire Change Detection")
st.write("Upload a satellite image to generate a change map.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Process Image
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Segmenter
    with torch.no_grad():
        seg_output = segmenter(image_tensor)
        seg_pred = torch.argmax(seg_output, dim=1).squeeze().cpu().numpy()
    
    # Map DeepGlobe classes to wildfire states
    # Load class_dict.csv (adjust path as needed)
    class_dict = pd.read_csv('/content/drive/MyDrive/WildfireProject/datasets/deepglobe/class_dict.csv')
    change_map = np.zeros((seg_pred.shape[0], seg_pred.shape[1], 3), dtype=np.uint8)
    for idx, row in class_dict.iterrows():
        color = tuple(row[['r', 'g', 'b']].values)
        if row['name'] == 'barren_land':
            change_map[seg_pred == idx] = [255, 255, 255]  # Burned: White
        elif row['name'] == 'forest_land':
            change_map[seg_pred == idx] = [0, 255, 0]      # Recovery: Green
        else:
            change_map[seg_pred == idx] = [0, 0, 0]        # Unburned/Other: Black
    
    # Display
    st.image(change_map, caption='Change Detection Map', use_column_width=True)
    
    # Save and Download
    plt.imsave('change_map.png', change_map)
    with open('change_map.png', 'rb') as f:
        st.download_button('Download Change Map (PNG)', f, file_name='change_map.png')