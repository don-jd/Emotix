
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import os
from flask import Flask, render_template, Response, jsonify
import threading
import time
import io
class ResNet50Arc(nn.Module):
    def __init__(self, num_classes, embedding_size=512):
        super(ResNet50Arc, self).__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity() 

        self.embedding = nn.Linear(2048, embedding_size) 
        self.bn = nn.BatchNorm1d(embedding_size)
        self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        x = self.bn(x)
        x = self.classifier(x)
        return x


app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"C:\Users\donjo\Desktop\Projects\MirrorAi\model\ResNet50Arc_2025-07-19_20-41-51_final_model.pt"
CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

image_transforms = transforms.Compose([
    transforms.Resize(256),            
    transforms.CenterCrop(224),       
    transforms.ToTensor(),             
    transforms.Normalize(              
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


camera = None
model = None
model_loaded_successfully = False
num_classes_from_model = None
latest_frame = None 
frame_lock = threading.Lock() 


def load_and_prepare_model(model_path, device):
    global model, model_loaded_successfully, num_classes_from_model
    print(f"\n--- Attempting to load model from: {model_path} ---")
    try:
        loaded_full_dict = torch.load(model_path, map_location=device, weights_only=False)
        print("Successfully loaded the full file content.")

        if isinstance(loaded_full_dict, dict):
            
            if 'num_classes' in loaded_full_dict:
                actual_num_classes = loaded_full_dict['num_classes']
                print(f"Detected num_classes from loaded file: {actual_num_classes}")
            else:
                
                actual_num_classes = len(CLASS_NAMES)
                print(f"Warning: 'num_classes' not found directly in checkpoint. Assuming {actual_num_classes} based on CLASS_NAMES length.")
            
            
            state_dict_to_load = None
            if "model_state_dict" in loaded_full_dict:
                nested_state_dict = loaded_full_dict['model_state_dict']
                print("Extracted 'model_state_dict' as the base dictionary.")
                cleaned_state_dict = {}
                for k, v in nested_state_dict.items():
                    if k.startswith('online_model.'):
                        new_key = k[len('online_model.'):]
                        cleaned_state_dict[new_key] = v
                state_dict_to_load = cleaned_state_dict
                print("Cleaned state_dict by removing 'online_model.' prefix and filtering extraneous keys.")
            elif "online_model" in loaded_full_dict:
                state_dict_to_load = loaded_full_dict['online_model']
                print("Extracted 'online_model' as the state_dict for loading.")
                
                state_dict_to_load = {k: v for k, v in state_dict_to_load.items() if k not in ["initted", "step"]}
            else:
                print("Loaded file is a dictionary, but neither 'model_state_dict' nor 'online_model' keys found.")
                print("Assuming the top-level dictionary IS the state_dict and filtering it.")
                state_dict_to_load = {k: v for k, v in loaded_full_dict.items() if not k.startswith('ema_model.') and k not in ["initted", "step"]}
        else:
            print("The loaded file appears to be the state_dict directly (not a dictionary).")
            state_dict_to_load = loaded_full_dict
           
            actual_num_classes = len(CLASS_NAMES)


        if state_dict_to_load is None:
            raise ValueError("Could not determine the state_dict to load from the file content.")
        if actual_num_classes is None:
            raise ValueError("Could not determine the number of classes from the loaded model file.")

        model = ResNet50Arc(num_classes=actual_num_classes).to(device)
        print("\n--- Attempting to load state_dict into model with strict=True ---")
        model.load_state_dict(state_dict_to_load, strict=True)
        model.eval() 
        print("Model loaded successfully with strict=True and set to eval mode!")
        model_loaded_successfully = True
        num_classes_from_model = actual_num_classes 
        
        
        if num_classes_from_model != len(CLASS_NAMES):
            print(f"ERROR: Model expects {num_classes_from_model} classes, but CLASS_NAMES has {len(CLASS_NAMES)} entries.")
            print("Please ensure your CLASS_NAMES list in app.py exactly matches the number of output classes of your trained model.")
            model_loaded_successfully = False 

    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Please check the path.")
        model_loaded_successfully = False
    except Exception as e:
        print(f"An unexpected error occurred during model loading: {e}")
        model_loaded_successfully = False

def capture_frames():
    global camera, latest_frame
    camera = cv2.VideoCapture(0) 
    if not camera.isOpened():
        print("Error: Could not open camera. Please check if camera is connected and not in use.")
        return


    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Failed to grab frame from camera. Exiting video feed.")
            break
        
        with frame_lock:
            latest_frame = frame.copy() 


        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        

        time.sleep(0.03) 

    camera.release()
    print("Camera released.")



@app.route('/')
def index():
    return render_template('index.html')
@app.route('/emotix')
def emotix():
    return render_template('emotix.html')

@app.route('/video_feed')
def video_feed():
    return Response(capture_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict_from_camera', methods=['POST'])
def predict_from_camera():
    global model, model_loaded_successfully, latest_frame, num_classes_from_model

    if not model_loaded_successfully:
        return jsonify({"error": "Model not loaded successfully. Please check server logs."}), 500

    if latest_frame is None:
        return jsonify({"error": "No frame available from camera. Please ensure camera is running."}), 500

    with frame_lock:
        frame_to_process = latest_frame.copy() 

    try:

        pil_image = Image.fromarray(cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB))
        

        input_tensor = image_transforms(pil_image).to(device)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad(): 
            output = model(input_batch)

 
        probabilities = torch.softmax(output, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class_idx].item() 

        if 0 <= predicted_class_idx < len(CLASS_NAMES):
            predicted_class_name = CLASS_NAMES[predicted_class_idx]
        else:
            predicted_class_name = f"Unknown Emotion (Index: {predicted_class_idx})"
            print(f"Warning: Predicted index {predicted_class_idx} is out of bounds for CLASS_NAMES.")

        return jsonify({
            "success": True,
            "predicted_class_index": predicted_class_idx,
            "predicted_class_name": predicted_class_name,
            "confidence": f"{confidence:.4f}" 
        })

    except Exception as e:
 
        print(f"Error during prediction: {e}")
        return jsonify({"error": f"An unexpected error occurred during prediction: {str(e)}"}), 500

if __name__ == '__main__':
   
    load_and_prepare_model(MODEL_PATH, device)

    if not model_loaded_successfully:
        print("Model failed to load. The application will not be able to perform predictions.")
        

   
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    

    if camera:
        camera.release()
        cv2.destroyAllWindows()
    print("Flask application stopped and camera resources released.")