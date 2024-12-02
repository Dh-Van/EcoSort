from ultralytics import YOLO

model = YOLO('trained.pt')
model.predict(
   source='metal.jpg',
   conf=0.25,
   save=True
)
