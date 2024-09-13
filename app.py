import cv2

import torch
from flask import Flask, render_template, Response
from skimage import img_as_ubyte
import numpy as np
import yaml
from PIL import Image
from torchvision import transforms



# Flask app initialization
app = Flask(__name__)

# Function to load the pre-trained model checkpoints
def load_checkpoints(config_path, checkpoint_path, cpu=False):
    with open(config_path) as f:
        config = yaml.full_load(f)

    if cpu:
        # Load model for CPU
        generator = torch.load(checkpoint_path, map_location=torch.device('cpu'))['generator']
        checkpoint = torch.load(checkpoint_path)
        print(checkpoint.keys())  # Print out available keys
        kp_detector = torch.load(checkpoint_path, map_location=torch.device('cpu'))['kp_detector']
    else:
        # Load model for GPU
        generator = torch.load(checkpoint_path)['generator']
        kp_detector = torch.load(checkpoint_path)['kp_detector']

    generator.eval()
    kp_detector.eval()
    return generator, kp_detector




# Load the pre-trained model (First Order Motion Model)
config_path = 'config/vox-265.yaml'  # Path to configuration file
checkpoint_path = 'checkpoints/vox-cpk.pth.tar'  # Path to checkpoint file
generator, kp_detector = load_checkpoints(config_path, checkpoint_path)

# Load the source image (neutral face)
source_image = cv2.imread('neutral_face.jpg')  # Make sure this path is correct
source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)

# Initialize OpenCV VideoCapture for webcam
cap = cv2.VideoCapture(0)


def make_animation(source_image, driving_frames, generator, kp_detector):
    # Preprocessing
    def preprocess_image(image):
        image = Image.fromarray(image)
        image = transforms.ToTensor()(image).unsqueeze(0)
        return image

    def postprocess_image(image):
        image = image.squeeze().cpu().detach().numpy()
        image = np.transpose(image, (1, 2, 0))
        return (image * 255).astype(np.uint8)

    # Convert source and driving images to tensors
    source_tensor = preprocess_image(source_image)
    driving_tensors = [preprocess_image(frame) for frame in driving_frames]

    # Move tensors to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    source_tensor = source_tensor.to(device)
    driving_tensors = [tensor.to(device) for tensor in driving_tensors]

    # Get keypoints for source image
    with torch.no_grad():
        source_kp = kp_detector(source_tensor)

    # Generate animations
    results = []
    for driving_tensor in driving_tensors:
        driving_kp = kp_detector(driving_tensor)
        prediction = generator(source_tensor, driving_tensor, source_kp, driving_kp)
        result_image = postprocess_image(prediction)
        results.append(result_image)

    return results


# Function to generate frames with real-time face swap
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame to match source image size
            driving_frame = cv2.resize(frame_rgb, (source_image.shape[1], source_image.shape[0]))
            
            # Perform deepfake animation (face swap)
            predictions = make_animation(source_image, [driving_frame], generator, kp_detector)
            
            # Convert the prediction back to OpenCV format
            result_image = img_as_ubyte(predictions[0])
            result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            
            # Encode the image as jpeg
            ret, buffer = cv2.imencode('.jpg', result_image_bgr)
            frame = buffer.tobytes()

            # Yield the frame in byte format for real-time display
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Flask route to render the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Flask route to provide real-time video feed with face swap
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Main entry point to run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
