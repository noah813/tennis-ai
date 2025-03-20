from ultralytics import YOLO

model = YOLO('../assets/models/yolo11x.pt')

result = model.predict('assets/videos/test.jpg', save=True)
print(result)
print('boxes:')
for box in result[0].boxes:
    print(box)  