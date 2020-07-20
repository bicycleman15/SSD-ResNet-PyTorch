import torch
from priorbox import PriorBox
from model import SSD300
from PIL import Image

from box_utils import detect_objects

# Set the device
device = config['device']

priors = PriorBox(config).create_priors().to(device)

# Set up model
model = SSD300()

# Load model here if you want
# torch.load('COCO-Resnet.pth')

model = model.to(device)

# Open the image
img_path = 'test.jpg'
original_image = Image.open(img_path, mode='r')
original_image = original_image.convert('RGB')

# Transform
image = normalize(to_tensor(resize(original_image)))

# Move to default device
image = image.to(device)

# Forward prop.
predicted_locs, predicted_scores = model(image.unsqueeze(0))

# Detect objects in SSD output
pred_boxes, pred_labels, pred_scores = detect_objects(predicted_locs, predicted_scores, priors, min_score=0.01,
                                                            max_overlap=0.45, top_k=200)

# Move detections to the CPU
pred_boxes = pred_boxes[0].to('cpu')

# Transform to original image dimensions
original_dims = torch.FloatTensor([original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
pred_boxes = pred_boxes * original_dims

# Now can plot det_boxes
# Use det labels to know the class of the object
