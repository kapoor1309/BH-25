import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  # Import your custom model
import torch.nn as nn

# Paths to model files
model_1_path = r"trained_model_fold_.pth"
model_2_path = r"multi_task_model.pth"


# Define the triplet dictionary
triplet_dict = {
    0: ('grasper', 'dissect', 'cystic_plate'),
    1: ('grasper', 'dissect', 'gallbladder'),
    2: ('grasper', 'dissect', 'omentum'),
    3: ('grasper', 'grasp', 'cystic_artery'),
    4: ('grasper', 'grasp', 'cystic_duct'),
    5: ('grasper', 'grasp', 'cystic_pedicle'),
    6: ('grasper', 'grasp', 'cystic_plate'),
    7: ('grasper', 'grasp', 'gallbladder'),
    8: ('grasper', 'grasp', 'gut'),
    9: ('grasper', 'grasp', 'liver'),
    10: ('grasper', 'grasp', 'omentum'),
    11: ('grasper', 'grasp', 'peritoneum'),
    12: ('grasper', 'grasp', 'specimen_bag'),
    13: ('grasper', 'pack', 'gallbladder'),
    14: ('grasper', 'retract', 'cystic_duct'),
    15: ('grasper', 'retract', 'cystic_pedicle'),
    16: ('grasper', 'retract', 'cystic_plate'),
    17: ('grasper', 'retract', 'gallbladder'),
    18: ('grasper', 'retract', 'gut'),
    19: ('grasper', 'retract', 'liver'),
    20: ('grasper', 'retract', 'omentum'),
    21: ('grasper', 'retract', 'peritoneum'),
    22: ('bipolar', 'coagulate', 'abdominal_wall_cavity'),
    23: ('bipolar', 'coagulate', 'blood_vessel'),
    24: ('bipolar', 'coagulate', 'cystic_artery'),
    25: ('bipolar', 'coagulate', 'cystic_duct'),
    26: ('bipolar', 'coagulate', 'cystic_pedicle'),
    27: ('bipolar', 'coagulate', 'cystic_plate'),
    28: ('bipolar', 'coagulate', 'gallbladder'),
    29: ('bipolar', 'coagulate', 'liver'),
    30: ('bipolar', 'coagulate', 'omentum'),
    31: ('bipolar', 'coagulate', 'peritoneum'),
    32: ('bipolar', 'dissect', 'adhesion'),
    33: ('bipolar', 'dissect', 'cystic_artery'),
    34: ('bipolar', 'dissect', 'cystic_duct'),
    35: ('bipolar', 'dissect', 'cystic_plate'),
    36: ('bipolar', 'dissect', 'gallbladder'),
    37: ('bipolar', 'dissect', 'omentum'),
    38: ('bipolar', 'grasp', 'cystic_plate'),
    39: ('bipolar', 'grasp', 'liver'),
    40: ('bipolar', 'grasp', 'specimen_bag'),
    41: ('bipolar', 'retract', 'cystic_duct'),
    42: ('bipolar', 'retract', 'cystic_pedicle'),
    43: ('bipolar', 'retract', 'gallbladder'),
    44: ('bipolar', 'retract', 'liver'),
    45: ('bipolar', 'retract', 'omentum'),
    46: ('hook', 'coagulate', 'blood_vessel'),
    47: ('hook', 'coagulate', 'cystic_artery'),
    48: ('hook', 'coagulate', 'cystic_duct'),
    49: ('hook', 'coagulate', 'cystic_pedicle'),
    50: ('hook', 'coagulate', 'cystic_plate'),
    51: ('hook', 'coagulate', 'gallbladder'),
    52: ('hook', 'coagulate', 'liver'),
    53: ('hook', 'coagulate', 'omentum'),
    54: ('hook', 'cut', 'blood_vessel'),
    55: ('hook', 'cut', 'peritoneum'),
    56: ('hook', 'dissect', 'blood_vessel'),
    57: ('hook', 'dissect', 'cystic_artery'),
    58: ('hook', 'dissect', 'cystic_duct'),
    59: ('hook', 'dissect', 'cystic_plate'),
    60: ('hook', 'dissect', 'gallbladder'),
    61: ('hook', 'dissect', 'omentum'),
    62: ('hook', 'dissect', 'peritoneum'),
    63: ('hook', 'retract', 'gallbladder'),
    64: ('hook', 'retract', 'liver'),
    65: ('scissors', 'coagulate', 'omentum'),
    66: ('scissors', 'cut', 'adhesion'),
    67: ('scissors', 'cut', 'blood_vessel'),
    68: ('scissors', 'cut', 'cystic_artery'),
    69: ('scissors', 'cut', 'cystic_duct'),
    70: ('scissors', 'cut', 'cystic_plate'),
    71: ('scissors', 'cut', 'liver'),
    72: ('scissors', 'cut', 'omentum'),
    73: ('scissors', 'cut', 'peritoneum'),
    74: ('scissors', 'dissect', 'cystic_plate'),
    75: ('scissors', 'dissect', 'gallbladder'),
    76: ('scissors', 'dissect', 'omentum'),
    77: ('clipper', 'clip', 'blood_vessel'),
    78: ('clipper', 'clip', 'cystic_artery'),
    79: ('clipper', 'clip', 'cystic_duct'),
    80: ('clipper', 'clip', 'cystic_pedicle'),
    81: ('clipper', 'clip', 'cystic_plate'),
    82: ('irrigator', 'aspirate', 'fluid'),
    83: ('irrigator', 'dissect', 'cystic_duct'),
    84: ('irrigator', 'dissect', 'cystic_pedicle'),
    85: ('irrigator', 'dissect', 'cystic_plate'),
    86: ('irrigator', 'dissect', 'gallbladder'),
    87: ('irrigator', 'dissect', 'omentum'),
    88: ('irrigator', 'irrigate', 'abdominal_wall_cavity'),
    89: ('irrigator', 'irrigate', 'cystic_pedicle'),
    90: ('irrigator', 'irrigate', 'liver'),
    91: ('irrigator', 'retract', 'gallbladder'),
    92: ('irrigator', 'retract', 'liver'),
    93: ('irrigator', 'retract', 'omentum'),
    94: ('grasper', 'null_verb', 'null_target'),
    95: ('bipolar', 'null_verb', 'null_target'),
    96: ('hook', 'null_verb', 'null_target'),
    97: ('scissors', 'null_verb', 'null_target'),
    98: ('clipper', 'null_verb', 'null_target'),
    99: ('irrigator', 'null_verb', 'null_target')
}


# MultiTaskModel definition
class MultiTaskModel(nn.Module):
    def __init__(self, num_triplets, num_tools, num_verbs, num_targets):
        super(MultiTaskModel, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        backbone_output_dim = 2048

        self.triplet_head = nn.Linear(backbone_output_dim, num_triplets)
        self.tool_head = nn.Linear(backbone_output_dim, num_tools)
        self.verb_head = nn.Linear(backbone_output_dim, num_verbs)
        self.target_head = nn.Linear(backbone_output_dim, num_targets)

    def forward(self, x):
        features = self.backbone(x)
        triplet_preds = self.triplet_head(features)
        tool_preds = self.tool_head(features)
        verb_preds = self.verb_head(features)
        target_preds = self.target_head(features)
        return triplet_preds, tool_preds, verb_preds, target_preds


# Load models
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Task 1 model
    model_1 = models.resnet50(pretrained=True)
    model_1.fc = torch.nn.Linear(model_1.fc.in_features, 6)
    model_1.load_state_dict(torch.load(model_1_path, map_location=device))
    model_1 = model_1.to(device)
    model_1.eval()

    # Task 2 model
    model_2 = MultiTaskModel(num_triplets=100, num_tools=6, num_verbs=10, num_targets=15)
    model_2.load_state_dict(torch.load(model_2_path, map_location=device))
    model_2 = model_2.to(device)
    model_2.eval()

    cam_extractor = GradCAM(model=model_1, target_layers=[model_1.layer4[-1]])

    return model_1, model_2, cam_extractor, device


# Image transformations
transform = transforms.Compose([
    transforms.Resize((256, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Main Streamlit app
st.title("Instrument Localization and Triplet Recognition")
st.write("Upload an image to see the bounding boxes and triplet predictions!")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    # Load models
    model_1, model_2, cam_extractor, device = load_models()

    # Preprocess and predict
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        triplet_preds, _, _, _ = model_2(input_tensor)
    recognition_probs = torch.sigmoid(triplet_preds).cpu().numpy()[0].tolist()

    # Generate CAM for detection
    grayscale_cam = cam_extractor(input_tensor=input_tensor, targets=[ClassifierOutputTarget(0)])
    cam = grayscale_cam[0]
    cam_resized = cv2.resize(cam, (image.width, image.height))

    # Generate bounding boxes
    threshold = 0.5
    binary_mask = (cam_resized >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes
    original_image = np.array(image)
    detections = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Get the predicted triplet for this detection
        predicted_triplet_id = np.argmax(recognition_probs)  # Select the most likely triplet
        predicted_triplet = triplet_dict.get(predicted_triplet_id, ('Unknown', 'Unknown', 'Unknown'))
        
        label_text = f'{predicted_triplet[0]} {predicted_triplet[1]} {predicted_triplet[2]}'
        cv2.putText(original_image, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        detections.append({"x": x, "y": y, "width": w, "height": h, "triplet": predicted_triplet})

    st.image(original_image, caption="Processed Image with Bounding Boxes", use_column_width=True)
    st.write("Detections:", detections)
    st.write("Triplet Recognition Probabilities:", recognition_probs)
