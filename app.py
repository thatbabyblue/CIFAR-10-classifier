import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import myCNN

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

@st.cache_resource
def load_model():
    model = myCNN()
    model.load_state_dict(torch.load("cifar10_model.pth"))
    model.eval()
    return model

def predict_image(img, model):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        _, predicted = output.max(1)
    return CLASS_NAMES[predicted.item()]

st.title("CIFAR-10 Image Classifier")
uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)
    model = load_model()
    label = predict_image(img, model)
    st.write(f"### Predicted: **{label}**")