import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern professional look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stAlert {
        background-color: #e7f3ff;
        border-left: 5px solid #2196F3;
    }
    h1 {
        color: #1e3a8a;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h2, h3 {
        color: #334155;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Constants
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
CLASS_COLORS = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6']
MODEL_PATH = 'checkpoints/best_model.pth'

# Model definition (must match training exactly)
def create_densenet121_model(num_classes=4):
    model = models.densenet121(pretrained=False)
    
    # Freeze first 3 dense blocks (same as training)
    for name, param in model.named_parameters():
        if any(x in name for x in ['conv0', 'denseblock1', 'denseblock2', 'denseblock3']):
            param.requires_grad = False
    
    # Classifier (same as training)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    
    return model

# Image preprocessing
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Load model with caching
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_densenet121_model(num_classes=4)
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error(f"❌ Model file not found at: {MODEL_PATH}")
        st.info("Please ensure 'best_model.pth' is in the 'checkpoints' folder.")
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

# Prediction function
def predict(image, model, device):
    transform = get_transform()
    
    # Convert grayscale to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Transform and add batch dimension
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Convert to numpy
    probs = probabilities.cpu().numpy()[0]
    pred_class = predicted.item()
    conf_score = confidence.item()
    
    return pred_class, conf_score, probs

# Create probability bar chart
def create_probability_chart(probabilities, class_names):
    # Sort by probability
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_probs = probabilities[sorted_indices]
    sorted_classes = [class_names[i] for i in sorted_indices]
    sorted_colors = [CLASS_COLORS[i] for i in sorted_indices]
    
    fig = go.Figure(data=[
        go.Bar(
            y=sorted_classes,
            x=sorted_probs * 100,
            orientation='h',
            marker=dict(
                color=sorted_colors,
                line=dict(color='rgba(0,0,0,0.3)', width=1)
            ),
            text=[f'{p*100:.2f}%' for p in sorted_probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Probability (%)",
        yaxis_title="Tumor Type",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        xaxis=dict(range=[0, 100], gridcolor='#e5e7eb'),
        yaxis=dict(gridcolor='#e5e7eb')
    )
    
    return fig

# Create confidence gauge
def create_confidence_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Score", 'font': {'size': 20}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#2196F3"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#fee2e2'},
                {'range': [50, 75], 'color': '#fef3c7'},
                {'range': [75, 90], 'color': '#dbeafe'},
                {'range': [90, 100], 'color': '#d1fae5'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'size': 12}
    )
    
    return fig

# Main app
def main():
    # Header
    st.markdown("<h1 style='text-align: center;'>🧠 Brain Tumor Classification System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b; font-size: 18px;'>AI-Powered MRI Analysis using Deep Learning</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/cotton/128/000000/brain--v1.png", width=100)
        st.markdown("## About")
        st.info("""
        This application uses a **DenseNet121** deep learning model trained on brain MRI scans to classify tumors into four categories:
        
        - 🔴 **Glioma**
        - 🟠 **Meningioma**
        - 🟢 **No Tumor**
        - 🔵 **Pituitary**
        
        **Model Performance:**
        - Test Accuracy: 98.40%
        - Training: 5,712 images
        - Architecture: DenseNet121
        """)
        
        st.markdown("---")
        st.markdown("### How to Use")
        st.markdown("""
        1. Upload a brain MRI scan (JPG/PNG)
        2. Click 'Analyze Image'
        3. View prediction results
        """)
        
        st.markdown("---")
        st.markdown("### System Status")
        device_info = "🟢 GPU Available" if torch.cuda.is_available() else "🟡 CPU Mode"
        st.markdown(f"**Device:** {device_info}")
        if torch.cuda.is_available():
            st.markdown(f"**GPU:** {torch.cuda.get_device_name(0)}")
    
    # Load model
    model, device = load_model()
    
    if model is None:
        st.stop()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload MRI Scan")
        uploaded_file = st.file_uploader(
            "Choose a brain MRI image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a grayscale or RGB MRI scan in JPG or PNG format"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded MRI Scan', use_column_width=True)
            
            # Image info
            st.markdown("**Image Details:**")
            st.text(f"Size: {image.size[0]} x {image.size[1]} pixels")
            st.text(f"Mode: {image.mode}")
            st.text(f"Format: {image.format}")
    
    with col2:
        st.markdown("### 🔬 Analysis Results")
        
        if uploaded_file is not None:
            if st.button("🚀 Analyze Image", type="primary", use_container_width=True):
                with st.spinner('Analyzing MRI scan...'):
                    # Make prediction
                    pred_class, confidence, probabilities = predict(image, model, device)
                    predicted_label = CLASS_NAMES[pred_class]
                    
                    # Display result
                    st.markdown("---")
                    
                    # Prediction result with color-coded box
                    result_color = CLASS_COLORS[pred_class]
                    st.markdown(f"""
                    <div style='background-color: {result_color}22; padding: 20px; border-radius: 10px; border-left: 5px solid {result_color};'>
                        <h2 style='margin: 0; color: {result_color};'>Prediction: {predicted_label}</h2>
                        <p style='margin: 10px 0 0 0; font-size: 18px;'>Confidence: {confidence*100:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Confidence gauge
                    st.plotly_chart(create_confidence_gauge(confidence), use_container_width=True)
        else:
            st.info("👆 Please upload an MRI scan to begin analysis")
    
    # Probability chart (full width)
    if uploaded_file is not None and 'probabilities' in locals():
        st.markdown("---")
        st.markdown("### 📊 Detailed Probability Distribution")
        
        # Display probabilities in metrics
        cols = st.columns(4)
        for idx, (class_name, prob) in enumerate(zip(CLASS_NAMES, probabilities)):
            with cols[idx]:
                st.metric(
                    label=class_name,
                    value=f"{prob*100:.2f}%",
                    delta=None
                )
        
        # Bar chart
        st.plotly_chart(create_probability_chart(probabilities, CLASS_NAMES), use_container_width=True)
        
        # Interpretation
        st.markdown("---")
        st.markdown("### 💡 Interpretation Guide")
        
        if confidence > 0.9:
            interpretation = "🟢 **High Confidence**: The model is very confident in this prediction."
        elif confidence > 0.75:
            interpretation = "🟡 **Moderate Confidence**: The prediction is likely correct, but consider additional validation."
        else:
            interpretation = "🔴 **Low Confidence**: The model is uncertain. Please consult a medical professional."
        
        st.info(interpretation)
        
        st.warning("⚠️ **Medical Disclaimer**: This tool is for educational and research purposes only. It should not replace professional medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.")

if __name__ == "__main__":
    main()
