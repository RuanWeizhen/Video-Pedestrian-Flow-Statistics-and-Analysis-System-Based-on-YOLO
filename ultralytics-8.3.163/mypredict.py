from sympy.printing.pretty.pretty_symbology import line_width
from torch.fx.experimental.unification.multipledispatch.dispatcher import source

from ultralytics import YOLO

model = YOLO(r"yolov8s.pt") #用yolo11n.pt这个模型预测

model.predict(
    source=r"ultralytics\assets",#预测文件里的所有图片
    save=True,#save=True->保存一下预测结果，对应save=False
    show=False,#show=False->不用立刻显示结果，对应show=True
    #line_width=8 线条宽度

)