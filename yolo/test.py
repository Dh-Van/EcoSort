from ultralytics import YOLO

model = YOLO('runs/detect/train5/weights/best.pt')
model.predict(
   source='metal.jpg',
   conf=0.25,
   save=True
)
