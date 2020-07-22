from torchvision.ops.boxes import batched_nms


def filter_boxes_batched(decoded_boxes, box_confs, box_labels, nms_thresh=0.45, topk=750, min_conf=0.05):
    """
    Filters boxes by applying nms and returning a list of size of batch_size
    with the filtered boxes.
    :param decoded_boxes: [num,8732,4] shape boxes in boundary coords
    :param box_confs: [num, 8732] shape
    :param box_labels: [num, 8732] shape
    :param nms_thresh: nms threshold to apply
    :param topk: keep top k preds only
    :param min_conf: min confidence to consider a valid prediction
    :return: filtered boxes in list
    """

    filtered_bboxes = []
    filtered_confs = []
    filtered_labels = []

    # batch size
    num = decoded_boxes.size(0)

    for i in range(num):
        dec_boxes = decoded_boxes[i]
        confs = box_confs[i]
        idxs = box_labels[i]

        # ignore low conf boxes
        mask = confs > min_conf
        dec_boxes = dec_boxes[mask]
        confs = confs[mask]
        idxs = idxs[mask]

        # apply nms now
        keep = batched_nms(dec_boxes, confs, idxs, nms_thresh)

        # now only keep topk only
        keep = keep[:topk]
        # now keep nms boxes only
        dec_boxes = dec_boxes[keep]
        confs = confs[keep]
        idxs = idxs[keep]

        filtered_bboxes.append(dec_boxes)
        filtered_confs.append(confs)
        filtered_labels.append(idxs)

    return filtered_bboxes, filtered_confs, filtered_labels
