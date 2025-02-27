
from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model='yolo11n.pt')

# 对单张图片进行目标检测
results = model("./ultralytics/assets/zidane.jpg")

# 可视化检测结果
results[0].show()

#性能评估
metrics = model.val()