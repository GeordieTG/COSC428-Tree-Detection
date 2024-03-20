from ultralytics import YOLO

model = YOLO('yolov8n.yaml')
results = model.train(data='yolov8tree.v2i.yolov5pytorch/data.yaml', epochs=5)