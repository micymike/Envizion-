import numpy as np 
import matplotlib.pyplot as plt
import insightface

from insightface.data import get_image as ins_get_image
from insightface.app import FaceAnalysis

# Step 1: Face detection
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

img = ins_get_image('t1')
plt.imshow(img[:, :, ::-1])
plt.show()

faces = app.get(img)

# Cropping and plotting the images
fig, axs = plt.subplots(1, len(faces), figsize=(12, 5))  # Dynamically adjust subplot count

for i, face in enumerate(faces):
    bbox = face.bbox  # Access bbox as an attribute
    bbox = [int(b) for b in bbox]
    axs[i].imshow(img[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1])
    axs[i].axis('off')

# Step 2: Face swapping
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False)

source_face = faces[0]
bbox = source_face.bbox
bbox = [int(b) for b in bbox]
plt.imshow(img[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1])
plt.show()

res = img.copy()
for face in faces:
    res = swapper.get(res, face, source_face, paste_back=True)

# Plot the swapped face
plt.imshow(res[:, :, ::-1])
plt.show()

res = []
for face in faces:
    _img, _ = swapper.get(img, face, source_face, paste_back=False)
    res.append(_img)
res = np.concatenate(res, axis=1)
fig, ax = plt.subplots(figsize=(15, 5))
ax.imshow(res[:, :, ::-1])
ax.axis('off')
plt.show()
