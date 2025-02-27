import os
from pathlib import Path
from ultralytics.utils.downloads import download
from PIL import Image
from tqdm import tqdm

def convert_box(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

def visdrone2yolo(dir):
    (dir / 'labels').mkdir(parents=True, exist_ok=True)  # 创建 labels 目录
    pbar = tqdm((dir / 'annotations').glob('*.txt'), desc=f'Converting {dir}')
    for f in pbar:
        img_size = Image.open((dir / 'images' / f.name).with_suffix('.jpg')).size
        lines = []
        with open(f, 'r') as file:  # 读取标注文件
            for row in [x.split(',') for x in file.read().strip().splitlines()]:
                if row[4] == '0':  # 忽略标注为 0 的区域
                    continue
                cls = int(row[5]) - 1
                box = convert_box(img_size, tuple(map(int, row[:4])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
                with open(str(f).replace(f'{os.sep}annotations{os.sep}', f'{os.sep}labels{os.sep}'), 'w') as fl:
                    fl.writelines(lines)  # 写入 YOLO 格式的标注

# 下载数据集
dir = Path("../datasets/VisDrone")  # 数据集根目录
urls = [
    'https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip',
    'https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip',
    'https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip',
    'https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-challenge.zip'
]
download(urls, dir=dir, curl=True, threads=4)

# 转换 VisDrone 标注格式到 YOLO 格式
for d in ['VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev']:
    visdrone2yolo(dir / d)
