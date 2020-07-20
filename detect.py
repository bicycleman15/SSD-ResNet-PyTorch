import torch
from model import SSD300

model = SSD300()

state_dict = torch.load('../COCO-resnet-2-COCO-version1-dev-val_loss-19.545.pth',map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

from PIL import Image,ImageDraw

original_image = Image.open('../val2017/000000001296.jpg', mode='r')
original_image = original_image.convert('RGB')

from utils import *
from torchvision import transforms

resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

image = normalize(to_tensor(resize(original_image)))

with torch.no_grad():
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

from box_utils import detect_objects
from priorbox import PriorBox
from config import config
priors = PriorBox(config).forward()

det_boxes, det_labels, det_scores = detect_objects(predicted_locs, predicted_scores, priors,min_score=0.5,max_overlap=0.5,top_k=200)

# Move detections to the CPU
det_boxes = det_boxes[0]

# Transform to original image dimensions
original_dims = torch.FloatTensor(
    [original_image.width, original_image.height, original_image.width, original_image.height])
det_boxes = det_boxes * original_dims

annotated_image = original_image
draw = ImageDraw.Draw(annotated_image)

# Suppress specific classes, if needed
for i in range(det_boxes.size(0)):

    # Boxes
    box_location = det_boxes[i].tolist()
    draw.rectangle(xy=box_location, outline='red')

del draw

print("Saving Image as test.jpg")
annotated_image.save(open('test.jpg','w'))