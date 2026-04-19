import torch
import pandas as pd

model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

imgs = ["https://ultralytics.com/images/zidane.jpg"]  # batch of images

model.conf = 0.30

# Inference
results = model(imgs)

# Results
results.print()
results.save()
results.show()
print(results.pandas().xyxy[0])
