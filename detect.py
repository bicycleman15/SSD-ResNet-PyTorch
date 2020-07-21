import torch
from model import SSD300

model = SSD300()

print("Loading Model...")
# state_dict = torch.load('../COCO-resnet-2-COCO-version1-dev-val_loss-19.545.pth',map_location='cpu')
# model.load_state_dict(state_dict)
model.eval()

from PIL import Image

from utils import *
from torchvision import transforms

from box_utils import detect_objects
from priorbox import PriorBox
from config import config
priors = PriorBox(config).forward()

def detect_objects_and_plot(original_image : 'PIL Image', model, priors, threshold = 0.5):
    """ Takes in an PIL image and model, run it on the image and returns an annotated PIL image. Also 
    takes in conf threshold above which to consider valid prediction.
    """
    # some transorms
    resize = transforms.Resize((300, 300))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    image = normalize(to_tensor(resize(original_image)))
    model.eval()
    image = image.unsqueeze(0)
    with torch.no_grad():
        predicted_locs, predicted_scores = model(image)
    det_boxes, det_labels, det_scores = detect_objects(predicted_locs, predicted_scores, priors, min_score=0.05,max_overlap=0.45,top_k=200)
    
    # Move detections to the CPU
    det_boxes = det_boxes[0]

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height])
    det_boxes = det_boxes * original_dims

    det_labels = det_labels[0]
    det_scores = det_scores[0]

    mask = det_scores > threshold

    det_boxes = det_boxes[mask]
    det_scores = det_scores[mask]
    det_labels = det_labels[mask]

    annotated_image = draw_boxes_(original_image, det_boxes, det_labels, det_scores)

    return annotated_image


if __name__ == '__main__':
    original_image = Image.open('../val2017/000000001296.jpg', mode='r')
    original_image = original_image.convert('RGB')
    annotated_image = detect_objects_and_plot(original_image, model, priors)
    print("Saving Image as test.jpg")
    annotated_image.save(open('test.jpg','w'))