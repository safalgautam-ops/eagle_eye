import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class DeepfakeModel(nn.Module):
    def __init__(self):
        super(DeepfakeModel, self).__init__()
        # Conv layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # First conv block
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        # Second conv block
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        # Third conv block
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x

class DeepfakeDetector:
    def __init__(self):
        # Create a PyTorch CNN model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DeepfakeModel().to(self.device)
        
        # In a real application, you would load pre-trained weights here
        # checkpoint = torch.load('deepfake_model_weights.pth')
        # self.model.load_state_dict(checkpoint)
        
    def extract_frames(self, video_path, num_frames=10):
        """Extract frames from a video file"""
        frames = []
        if not os.path.exists(video_path):
            print(f"Error: File {video_path} does not exist")
            return frames
            
        video = cv2.VideoCapture(video_path)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            print(f"Error: No frames in video {video_path}")
            return frames
            
        # Extract evenly spaced frames
        indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
        
        for i in indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.read()
            if ret:
                # Resize frame
                frame = cv2.resize(frame, (128, 128))
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                
        video.release()
        return frames
        
    def preprocess_frame(self, frame):
        """Preprocess a frame for the model"""
        # Convert to PyTorch tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(frame)
        
    def predict(self, video_path):
        """Predict if a video is real or deepfake"""
        frames = self.extract_frames(video_path)
        if not frames:
            return {"error": "Could not extract frames from video"}
            
        # Preprocess frames
        processed_frames = [self.preprocess_frame(frame) for frame in frames]
        
        # For demonstration purposes, since we don't have a trained model,
        # we'll return a random prediction with confidence
        # In a real application, you would use:
        # with torch.no_grad():
        #     batch = torch.stack(processed_frames).to(self.device)
        #     predictions = self.model(batch)
        #     avg_prediction = predictions.mean().item()
        
        # Simulating model prediction
        prediction = np.random.random()
        
        result = {
            "prediction": "FAKE" if prediction > 0.5 else "REAL",
            "confidence": float(prediction) if prediction > 0.5 else float(1 - prediction),
            "frames_analyzed": len(frames)
        }
        
        return result

if __name__ == "__main__":
    detector = DeepfakeDetector()
    # Test with an example video if available
    test_video = "../test_video.mp4"
    if os.path.exists(test_video):
        result = detector.predict(test_video)
        print(result)
    else:
        print(f"Test video not found: {test_video}")