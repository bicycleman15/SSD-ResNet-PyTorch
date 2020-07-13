# Implementing SSD 300--(Input image shape=[300,300,n]) where n is the number of channels

# We are following NVIDIA's approach where in
# --Using Resnet50 backbone
# --USing only first 4 residual layers of the Resnet ie dircrding conv5_x
#     and onward.
# -- All strides of conv4_x are made 1x1
# -- Addition of additional layers, same implementation as in the Resnet Paper

# Importing necessary modules
import torch
import torch.nn as nn  # for nn layer generation and combinations
# We will be using the resnet 50 model for our implementation
from torchvision.models.resnet import resnet50


class ResNet(nn.Module):
    # The resnet backbone is to be used as a feature provider.
    # For 300x300 we expect a 38x38 feature map

    # TODO: Add pretrained loading feature
    def __init__(self):
        super().__init__()
        # Loading the full Resnet 50 backbone. This means that the we are currently only
        # importing the REsnet50s architecture. No weights or biases have been initialised.
        # WE will be using Xavier initialisation for that
        # Loading the full Resnet 50 backbone
        backbone = resnet50(pretrained=False)

        # Extracting the the required layers form the backbone
        # nn.sequential converts the individial components extracted from resnet in a list to
        # to continiuos nn object on which we can perform backprop
        # Lets call this as our feautre provider. This provied us with the very first feature map [38x38] with 1024 channels
        self.feature_provider = nn.Sequential(*list(backbone.children())[:7])

        # NOTE: But it is necessary to change the layer's stride else the feature provider will give a feature of 19x19
        # The conv4_x layer is the last object in our feature provider list.
        # Since stride arvariable in only the first block of a resnet layer we select the
        # last layer with the [-1] index and its first block with [0] in the self.feature_provider[-1][0]
        conv4_block1 = self.feature_provider[-1][0]

        conv4_block1.conv1.stride = (1, 1)  # changing the stride to 1x1
        conv4_block1.conv2.stride = (1, 1)  # changing the stride to 1x1
        conv4_block1.downsample[0].stride = (
            1, 1)  # changing the stride to 1x1

    def forward(self, x):
        # provides a feature map in a forward pass
        x = self.feature_provider(x)
        return x  # [38,38,1024]


class SSD300(nn.Module):
    # Contains the full SSD300 model with additoinal layers and classification and localisation heads
    def __init__(self, backbone=ResNet()):
        super().__init__()

        self.feature_provider = backbone  # initialising our feature provider backbone
        self.label_num = 81  # number of COCO classes

        # contains all the feature maps's shapes as string for easy reference
        features_list = ["38x38", "19x19", "10x10", "5x5", "3x3", "1x1"]

        # a dictionary mapping having keys as the feature map shape as string
        feature_channel_dict = {"38x38": 1024,
                                "19x19": 512,
                                "10x10": 512,
                                "5x5": 256,
                                "3x3": 256,
                                "1x1": 256}

        # number of proposed priors per feauture in a feature map as given in the paper
        num_proir_box_dict = {"38x38": 4,
                              "19x19": 6,
                              "10x10": 6,
                              "5x5": 6,
                              "3x3": 4,
                              "1x1": 4}

        intermediate_channel_dict = {"19x19": 256,
                                     "10x10": 256,
                                     "5x5": 128,
                                     "3x3": 128,
                                     "1x1": 128}

        # intermediate channels for the additional layers
        self._make_additional_features_maps(
            features_list, feature_channel_dict, intermediate_channel_dict)

        self.loc = []
        self.conf = []

        # Generating localisatin heads and classification heads
        for feature_map_name in features_list:
            priors_boxes=num_proir_box_dict[feature_map_name]
            output_channel_from_feature_map=feature_channel_dict[feature_map_name]
            self.loc.append(nn.Conv2d(output_channel_from_feature_map, priors_boxes * 4, kernel_size=3, padding=1))
            self.conf.append(nn.Conv2d(output_channel_from_feature_map, priors_boxes * self.label_num, kernel_size=3, padding=1))

        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)
        self._init_weights()

    def _make_additional_features_maps(self, features_list, feature_channel_dict, intermediate_channel_dict):

        # input for additional layers come from one behinf it is coming from the
        input_list = features_list[:-1],
        # output is as stated for each additional layer
        output_list = features_list[1:],

        self.additional_blocks = []

        for i, (prev_feature_name, current_feature_name) in enumerate(zip(input_list, output_list)):
            if i < 3:  # for the first 3 additional features maps (19x19 , 10x10, 5x5) we use padding of 1 and stride of 2 for  
                       # the second convolution that generates additional map
                layer = nn.Sequential(
                    nn.Conv2d(feature_channel_dict[prev_feature_name],
                              intermediate_channel_dict[current_feature_name], kernel_size=1, bias=False),

                    
                    nn.BatchNorm2d(intermediate_channel_dict[current_feature_name]),  # an additional implementation by NVIDIA
                    nn.ReLU(inplace=True),

                    # the differentiating conv from other two (1x1 and 3x3) 
                    nn.Conv2d(intermediate_channel_dict[current_feature_name], feature_channel_dict[current_feature_name], kernel_size=3,
                              padding=1, stride=2, bias=False),
                    
                    nn.BatchNorm2d(feature_channel_dict[current_feature_name]),  # an additional implementation by NVIDIA
                    nn.ReLU(inplace=True),
                )
            else: # for the last additional features maps (3x3 and 1x1) we use padding of 0 and stride of 1 for  
                  # the second convolution that generates additional map
                layer = nn.Sequential(
                    nn.Conv2d(feature_channel_dict[prev_feature_name],
                              intermediate_channel_dict[current_feature_name], kernel_size=1, bias=False),
                    
                    nn.BatchNorm2d(intermediate_channel_dict[current_feature_name]),  # an additional implementation by NVIDIA
                    nn.ReLU(inplace=True),

                    # the differentiating conv from other three (19x19 , 10x10, 5x5)
                    nn.Conv2d(intermediate_channel_dict[current_feature_name], feature_channel_dict[current_feature_name],
                              kernel_size=3, bias=False),
                    
                    nn.BatchNorm2d(feature_channel_dict[current_feature_name]),  # an additional implementation by NVIDIA
                    nn.ReLU(inplace=True),
                )
            # adding the new feature map generator block to our arsenal    
            self.additional_blocks.append(layer)


        # converting into nn modules so that they can be added to pytorchs
        # computational graph and backprop can be performed
        self.additional_blocks = nn.ModuleList(self.additional_blocks)


    # Xavier initialising the weights
    def _init_weights(self):
        # making a list of all blocks in out SSD300 models
        # note that the backbone already has weights initialised so we are 
        # initialising only the newly created layer's weights
        layers = [*self.additional_blocks, *self.loc, *self.conf] # *list 
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    # Shape the classifier to the view of bboxes
    def bbox_view(self, src, loc, conf):
        ret = []
        for s, l, c in zip(src, loc, conf):
            ret.append((l(s).view(s.size(0), 4, -1),
                        c(s).view(s.size(0), self.label_num, -1)))

        locs, confs = list(zip(*ret))
        locs, confs = torch.cat(locs, 2).contiguous(
        ), torch.cat(confs, 2).contiguous()
        return locs, confs

    def forward(self, x):
        x = self.feature_provider(x)

        detection_feed = [x]
        for l in self.additional_blocks:
            x = l(x)
            detection_feed.append(x)

        # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)

        # For SSD 300, shall return nbatch x 8732 x {nlabels, nlocs} results
        return locs, confs


class Loss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """

    def __init__(self, dboxes):
        super(Loss, self).__init__()
        self.scale_xy = 1.0/dboxes.scale_xy
        self.scale_wh = 1.0/dboxes.scale_wh

        self.sl1_loss = nn.SmoothL1Loss(reduce=False)
        self.dboxes = nn.Parameter(dboxes(order="xywh").transpose(0, 1).unsqueeze(dim=0),
                                   requires_grad=False)
        # Two factor are from following links
        # http://jany.st/post/2017-11-05-single-shot-detector-ssd-from-scratch-in-tensorflow.html
        self.con_loss = nn.CrossEntropyLoss(reduce=False)

    def _loc_vec(self, loc):
        """
            Generate Location Vectors
        """
        gxy = self.scale_xy * \
            (loc[:, :2, :] - self.dboxes[:, :2, :])/self.dboxes[:, 2:, ]
        gwh = self.scale_wh*(loc[:, 2:, :]/self.dboxes[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def forward(self, ploc, plabel, gloc, glabel):
        """
            ploc, plabel: Nx4x8732, Nxlabel_numx8732
                predicted location and labels

            gloc, glabel: Nx4x8732, Nx8732
                ground truth location and labels
        """
        mask = glabel > 0
        pos_num = mask.sum(dim=1)

        vec_gd = self._loc_vec(gloc)

        # sum on four coordinates, and mask
        sl1 = self.sl1_loss(ploc, vec_gd).sum(dim=1)
        sl1 = (mask.float()*sl1).sum(dim=1)

        # hard negative mining
        con = self.con_loss(plabel, glabel)

        # postive mask will never selected
        con_neg = con.clone()
        con_neg[mask] = 0
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)

        # number of negative three times positive
        neg_num = torch.clamp(3*pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = con_rank < neg_num

        #print(con.shape, mask.shape, neg_mask.shape)
        closs = (con*(mask.float() + neg_mask.float())).sum(dim=1)

        # avoid no object detected
        total_loss = sl1 + closs
        num_mask = (pos_num > 0).float()
        pos_num = pos_num.float().clamp(min=1e-6)
        ret = (total_loss*num_mask/pos_num).mean(dim=0)
        return ret
