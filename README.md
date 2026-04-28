# Video Pedestrian Flow Statistics and Analysis System Based on YOLO

一个基于 YOLO + DeepSORT 的行人流量统计与分析系统，面向视频场景中的行人检测、跟踪、计数、客流趋势分析与隐私保护展示。

## 项目简介

本项目以 YOLO 目标检测为核心，结合多目标跟踪、区域/越线统计、静态目标过滤、轨迹可视化与热力图分析，实现对行人流量的持续统计和结果输出。系统支持视频文件或摄像头输入，并可生成统计结果、事件记录、趋势图和输出视频。

## 主要功能

- 行人检测：基于 YOLO 模型识别人像目标
- 多目标跟踪：结合 DeepSORT 为目标分配稳定 ID
- 越线统计：统计行人上行、下行及总流量
- 区域统计：支持自定义 ROI 与统计区域
- 静态目标过滤：减少长期停留目标对统计的干扰
- 轨迹与热力图：可视化运动轨迹和人流聚集区域
- 结果导出：输出视频、事件 CSV、汇总 JSON、流量趋势图
- 隐私保护：支持人脸模糊与可选加密
- GUI 运行：支持图形界面启动与交互式配置

## 系统特点

- 面向实际视频监控与客流分析场景
- 可通过 YAML 配置快速切换模型、视频源和统计策略
- 兼容 CPU / GPU 运行环境
- 输出结果结构清晰，便于二次分析和展示

## 目录结构

```text
README.md
.gitignore
ultralytics-8.3.163/
  pedestrian_system/
    main.py
    config/
    counting/
    detector/
    gui/
    privacy/
    tracker/
    utils/
    videos/
    tools/
```

## 快速开始

### 1. 环境准备

建议使用 Python 3.10 及以上版本，并安装项目所需依赖。

### 2. 启动 GUI

进入系统目录后运行：

```bash
python main.py --gui
```

### 3. 直接运行主程序

如果你希望使用配置文件直接处理视频，可以运行：

```bash
python main.py --config config/pedestrian_demo.yaml
```

## 配置说明

默认配置位于 [ultralytics-8.3.163/pedestrian_system/config/pedestrian_demo.yaml](ultralytics-8.3.163/pedestrian_system/config/pedestrian_demo.yaml)。

你可以在其中修改以下内容：

- `model_path`：YOLO 权重路径
- `source`：输入视频或摄像头编号
- `output_dir`：输出目录
- `counting`：越线统计与区域统计配置
- `static_filter`：静态目标过滤参数
- `privacy`：人脸模糊和加密选项

## 输出内容

程序运行后通常会生成以下结果：

- `result.mp4`：带可视化结果的视频
- `events.csv`：事件记录
- `summary.json`：统计汇总
- `flow.csv`：按分钟的客流统计
- `flow_trend.png`：客流趋势图

## 项目亮点

- 适合课堂展示、课程设计和毕业设计汇报
- 能直观展示行人检测、跟踪和统计流程
- 兼顾分析能力与隐私保护能力
- 输出文件便于做论文图表和结果截图

## 注意事项

- `model_path` 指向的权重文件必须存在，否则系统无法启动检测流程
- 若使用 GPU 但本机没有 CUDA，程序会自动回退到 CPU
- 视频源路径错误或摄像头不可用时，程序会报错并停止
- 统计区域和越线坐标需要根据实际画面重新调整

## 推荐展示方式

如果你准备把项目放到 GitHub 首页，建议配合以下内容一起展示：

- 一张 GUI 截图
- 一张检测结果截图
- 一张客流趋势图
- 一段 20 到 30 秒的演示视频

## 许可证

本项目基于 Ultralytics YOLO 生态进行开发。请在正式发布或商用前确认相关模型权重与依赖的许可证要求。
