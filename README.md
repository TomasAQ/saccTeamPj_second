# SSAC 2차 프로젝트 
## Yolov5 + Deep Sort with PyTorch

## 소개
차량과 차선을 인식하고 화면 하단에서 차량까지에 거리를 계산하는 프로그램


1. kitti dataset 분석 및 데이터 활용  
   yolo_tools/analysis_label.py : kitti dataset 분석을 위한 코드   
   yolo_tools/MakeYOLOLabel.py : yolov5 학습을 위한 라벨데이터 생성 코드   
2. Yolov5 모델 학습   
   yolov5/learning_yolo5.ipynb : yolov5 학습 코드
3. 차량과 차선 인식 및 거리 계산 모델(yolov5 + deepsort) 
    


### source
    https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch
    https://github.com/ultralytics/yolov5