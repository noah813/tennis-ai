import pandas as pd
import matplotlib.pyplot as plt

# 讀取CSV文件
csv_path = 'yolov11/training/runs/detect/train/results.csv'  # 修正路徑
data = pd.read_csv(csv_path)

# 繪製 cost function
plt.figure(figsize=(10, 6))
plt.plot(data['epoch'], data['train/box_loss'], label='Train Box Loss')
plt.plot(data['epoch'], data['val/box_loss'], label='Validation Box Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Cost Function')
plt.legend()
plt.grid(True)
plt.show()
