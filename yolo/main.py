from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Train the model
train_results = model.train(
    data="data.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=416,  # training image size
    device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("test/images/metal4_jpg.rf.95bf1c428ead5f8f591d5597fd3f81a6.jpg")
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported modelcl