from lightning.lightning_model import SSD300_COCO
import torch
from PIL import Image

from torchvision import transforms
from models.box_utils import decode
from models.detect_utils import filter_boxes_batched

from omegaconf import OmegaConf
import torch.nn.functional as F
from vizer.draw import draw_boxes

from dataset_coco.utils import coco_class_name
from dataset_coco.dataset import COCODataset


def detect_objects_and_plot(img: 'PIL Image', model, threshold=0.3):
    """ Takes in an PIL image and model, run it on the image and returns an annotated PIL image. Also
    takes in conf threshold above which to consider valid prediction.
    """
    # some transorms
    resize = transforms.Resize((300, 300))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    image = normalize(to_tensor(resize(img)))

    model.eval()
    with torch.no_grad():
        locs, confs = model(image.unsqueeze(0))

    for i in range(locs.size(0)):
        # Decode targets here if possible
        locs[i] = decode(locs[i], model.criterion.priors, model.cfg.priors.variance)

    confs = F.softmax(confs, dim=2)
    scores, idxs = confs.max(dim=2)

    bboxes, scores, labels = filter_boxes_batched(locs, scores, idxs, min_conf=threshold, nms_thresh=0.1)

    # since there is only a single image
    bboxes = bboxes[0]
    scores = scores[0]
    labels = labels[0]

    # Transform to original image dimensions
    dims = torch.FloatTensor([img.width, img.height, img.width, img.height])
    bboxes = bboxes * dims

    annotated_image = draw_boxes(img, bboxes, labels, scores, class_name_map=coco_class_name)
    annotated_image = Image.fromarray(annotated_image)
    return annotated_image

if __name__ == '__main__':
    # Set seed
    # set_seed()
    #
    # # parse config
    config = OmegaConf.load('config.yaml')
    #
    # # init model first
    # model = SSD300_COCO(cfg=config)

    data_train = COCODataset(config.data.val_data_path, config.data.val_annotate_path,'TEST')

    for img, bboxes, labels in data_train:
        print(img.shape)
        print(bboxes)
        print(labels)
        break

    # state_dict = torch.load('../epoch=24.ckpt', map_location='cpu')
    # model.load_state_dict(state_dict['state_dict'])
    #
    # img_raw = Image.open('../val2017/000000001503.jpg', mode='r').convert('RGB')
    #
    # ann_image = detect_objects_and_plot(img_raw, model)
    # print("Saving Image as test.jpg")
    # ann_image.save(open('test.jpg', 'w'))
