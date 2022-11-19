import matplotlib.pyplot as plt
import wandb

plt.style.use('ggplot')


def log_bboxes(image, bboxes, scores, class_labels):
    display_ids = {"back ground": 0}
    for idx, cls_lbl in enumerate(class_labels):
        display_ids[cls_lbl] = idx + 1

    class_id_to_label = {int(v): k for k, v in display_ids.items()}
    # load raw input photo
    all_bboxes = []
    # plot each bounding box for this image
    for idx, bbox in enumerate(bboxes):
        # get coordinates and labels
        bbox_data = {"position": {
          "minX": int(bbox[0]),
          "maxX": int(bbox[2]),
          "minY": int(bbox[1]),
          "maxY": int(bbox[3]),
          },
          "class_id": 1,
          # optionally caption each box with its class and score
          "box_caption": f"Yeast ({scores[idx]:.2f})",
          "domain": "pixel",
          "scores": {"score": float(scores[idx])}}
        all_bboxes.append(bbox_data)

    # log to wandb: raw image, predictions, and dictionary of class labels for each class id
    box_image = wandb.Image(image, boxes={"predictions": {"box_data": all_bboxes, "class_labels": class_id_to_label}})
    return box_image
