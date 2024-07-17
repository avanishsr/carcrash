from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
import os

from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
import base64

from detectron2.utils.visualizer import Visualizer, ColorMode

app = Flask(__name__)

# Register COCO instances
dataset_dir = "good"
img_dir = "img/"
val_dir = "val/"

register_coco_instances("car_dataset_val", {}, os.path.join(dataset_dir, val_dir, "COCO_val_annos.json"), os.path.join(dataset_dir, img_dir))
register_coco_instances("car_mul_dataset_val", {}, os.path.join(dataset_dir, val_dir, "COCO_mul_val_annos.json"), os.path.join(dataset_dir, img_dir))

# Load damage detection model
cfg_damage = get_cfg()
cfg_damage.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg_damage.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Only one class (damage) + background
cfg_damage.MODEL.WEIGHTS = "damage_segmentation_model.pth"
cfg_damage.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg_damage.MODEL.DEVICE = 'cpu'
damage_predictor = DefaultPredictor(cfg_damage)

# Load parts detection model
cfg_parts = get_cfg()
cfg_parts.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg_parts.MODEL.ROI_HEADS.NUM_CLASSES = 6  # Only five classes (headlamp, hood, rear_bumper, front_bumper, door) + background
cfg_parts.MODEL.WEIGHTS = "part_segmentation_model.pth"
cfg_parts.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg_parts.MODEL.DEVICE = 'cpu'
part_predictor = DefaultPredictor(cfg_parts)

damage_class_map = {0: 'damage'}
parts_class_map = {0: 'headlamp', 1: 'rear_bumper', 2: 'door', 3: 'hood', 4: 'front_bumper'}

def detect_damage_part(damage_dict, parts_dict):
    from scipy.spatial import distance
    try:
        max_distance = 10e9
        assert len(damage_dict) > 0, "AssertError: damage_dict should have at least one damage"
        assert len(parts_dict) > 0, "AssertError: parts_dict should have at least one part"
        max_distance_dict = dict(zip(damage_dict.keys(), [max_distance] * len(damage_dict)))
        part_name = dict(zip(damage_dict.keys(), [''] * len(damage_dict)))

        for y in parts_dict.keys():
            for x in damage_dict.keys():
                dis = distance.euclidean(damage_dict[x], parts_dict[y])
                if dis < max_distance_dict[x]:
                    part_name[x] = y.rsplit('_', 1)[0]

        return list(set(part_name.values()))
    except Exception as e:
        print(e)
        return []

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Damage inference
    damage_outputs = damage_predictor(img)

    # Visualization of damage predictions
    v = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("car_dataset_val"), scale=0.5)
    out = v.draw_instance_predictions(damage_outputs["instances"].to("cpu"))
    annotated_image = out.get_image()[:, :, ::-1]  # Convert RGB to BGR for OpenCV if needed

    # Part inference
    parts_outputs = part_predictor(img)
    parts_prediction_classes = [parts_class_map[el] + "_" + str(indx) for indx, el in enumerate(parts_outputs["instances"].pred_classes.tolist())]
    parts_polygon_centers = parts_outputs["instances"].pred_boxes.get_centers().tolist()
    parts_dict = dict(zip(parts_prediction_classes, parts_polygon_centers))

    parts_outputs = part_predictor(img)
    parts_v = Visualizer(img[:, :, ::-1],
                         metadata=MetadataCatalog.get("car_mul_dataset_val"),
                         scale=0.5,
                         instance_mode=ColorMode.IMAGE_BW
                         # remove the colors of unsegmented pixels. This option is only available for segmentation models
                         )
    parts_out = parts_v.draw_instance_predictions(parts_outputs["instances"].to("cpu"))
    annotated_image2 = parts_out.get_image()[:, :, ::-1]

    # Detect damaged parts
    damage_prediction_classes = [damage_class_map[el] + "_" + str(indx) for indx, el in
                                 enumerate(damage_outputs["instances"].pred_classes.tolist())]
    damage_polygon_centers = damage_outputs["instances"].pred_boxes.get_centers().tolist()
    damage_dict = dict(zip(damage_prediction_classes, damage_polygon_centers))
    # Detect damaged parts
    damaged_parts = detect_damage_part(damage_dict, parts_dict)

    #for 2nd
    _, img_bytes2 = cv2.imencode('.jpg', annotated_image2)
    annotated2_image_base64 = base64.b64encode(img_bytes2).decode('utf-8')



    # Convert annotated image to base64 string
    _, img_bytes = cv2.imencode('.jpg', annotated_image)
    annotated_image_base64 = base64.b64encode(img_bytes).decode('utf-8')

    return jsonify({'damaged_parts': damaged_parts, 'damage_segmentation': annotated_image_base64, 'part_segmentation': annotated2_image_base64 })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
