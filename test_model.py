from ultralytics import YOLO

# Load the PPE model
model = YOLO("models/best.pt")

# Print all classes in the model
print(model.names)