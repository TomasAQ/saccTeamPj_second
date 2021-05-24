import sys
sys.path.insert(0, './yolov5')

from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

import numpy as np

# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized 발생 제거
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

#############################################################
frames = 0
first_frame = 1
# Color
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
white = (255, 255, 255)
yellow = (0, 255, 255)
deepgray = (43, 43, 43)
dark = (1, 1, 1)
cyan = (255, 255, 0)
magenta = (255, 0, 255)
lime = (0, 255, 128)
# Global 함수 초기화
l_pos, r_pos, l_cent, r_cent = 0, 0, 0, 0
uxhalf, uyhalf, dxhalf, dyhalf = 0, 0, 0, 0
l_center, r_center, lane_center = ((0, 0)), ((0, 0)), ((0, 0))
next_frame = (0, 0, 0, 0, 0, 0, 0, 0)


############################################################################################

def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        # 상단에 라벨
        # cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        # cv2.putText(img, label, (x1, y1 +t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

################################################################################################################추가
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def get_pts(flag=0):
    vertices1 = np.array([
        [230, 650],
        [620, 460],
        [670, 460],
        [1050, 650]
    ])

    vertices2 = np.array([
        [0, 720],
        [710, 400],
        [870, 400],
        [1280, 720]
    ])
    if flag == 0: return vertices1
    if flag == 1: return vertices2


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # vertiecs로 만든 polygon으로 이미지의 ROI를 정하고 ROI 이외의 영역은 모두 검정색으로 정한다.

    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    # cv2.imshow('r', img)

    return line_img


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)


def get_slope(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1)


def draw_lines(img, lines):
    global cache
    global first_frame
    global next_frame

    y_global_min = img.shape[0]
    y_max = img.shape[0]

    l_slope, r_slope = [], []
    l_lane, r_lane = [], []

    det_slope = 0.5
    α = 0.2
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = get_slope(x1, y1, x2, y2)
                if slope > det_slope:
                    r_slope.append(slope)
                    r_lane.append(line)
                elif slope < -det_slope:
                    l_slope.append(slope)
                    l_lane.append(line)

        y_global_min = min(y1, y2, y_global_min)

    if (len(l_lane) == 0 or len(r_lane) == 0):  # 오류 방지
        return 1

    l_slope_mean = np.mean(l_slope, axis=0)
    r_slope_mean = np.mean(r_slope, axis=0)
    l_mean = np.mean(np.array(l_lane), axis=0)
    r_mean = np.mean(np.array(r_lane), axis=0)

    if ((r_slope_mean == 0) or (l_slope_mean == 0)):
        print('dividing by zero')
        return 1

    # y=mx+b -> b = y -mx
    l_b = l_mean[0][1] - (l_slope_mean * l_mean[0][0])
    r_b = r_mean[0][1] - (r_slope_mean * r_mean[0][0])

    if np.isnan((y_global_min - l_b) / l_slope_mean) or \
            np.isnan((y_max - l_b) / l_slope_mean) or \
            np.isnan((y_global_min - r_b) / r_slope_mean) or \
            np.isnan((y_max - r_b) / r_slope_mean):
        return 1

    l_x1 = int((y_global_min - l_b) / l_slope_mean)
    l_x2 = int((y_max - l_b) / l_slope_mean)
    r_x1 = int((y_global_min - r_b) / r_slope_mean)
    r_x2 = int((y_max - r_b) / r_slope_mean)

    if l_x1 > r_x1:  # Left line이 Right Line보다 오른쪽에 있는 경우 (Error)
        l_x1 = ((l_x1 + r_x1) / 2)
        r_x1 = l_x1

        l_y1 = ((l_slope_mean * l_x1) + l_b)
        r_y1 = ((r_slope_mean * r_x1) + r_b)
        l_y2 = ((l_slope_mean * l_x2) + l_b)
        r_y2 = ((r_slope_mean * r_x2) + r_b)

    else:  # l_x1 < r_x1 (Normal)
        l_y1 = y_global_min
        l_y2 = y_max
        r_y1 = y_global_min
        r_y2 = y_max

    current_frame = np.array([l_x1, l_y1, l_x2, l_y2, r_x1, r_y1, r_x2, r_y2], dtype="float32")

    if first_frame == 1:
        next_frame = current_frame
        first_frame = 0
    else:
        prev_frame = cache
        next_frame = (1 - α) * prev_frame + α * current_frame

    global l_center
    global r_center
    global lane_center

    div = 2
    l_center = (int((next_frame[0] + next_frame[2]) / div), int((next_frame[1] + next_frame[3]) / div))
    r_center = (int((next_frame[4] + next_frame[6]) / div), int((next_frame[5] + next_frame[7]) / div))
    lane_center = (int((l_center[0] + r_center[0]) / div), int((l_center[1] + r_center[1]) / div))

    global uxhalf, uyhalf, dxhalf, dyhalf
    uxhalf = int((next_frame[2] + next_frame[6]) / 2)
    uyhalf = int((next_frame[3] + next_frame[7]) / 2)
    dxhalf = int((next_frame[0] + next_frame[4]) / 2)
    dyhalf = int((next_frame[1] + next_frame[5]) / 2)

    cv2.line(img, (next_frame[0], next_frame[1]), (next_frame[2], next_frame[3]), red, 2)
    cv2.line(img, (next_frame[4], next_frame[5]), (next_frame[6], next_frame[7]), red, 2)

    cache = next_frame


def process_image(image):
    height, width = image.shape[:2]

    kernel_size = 3

    # Canny Edge Detection Threshold
    low_thresh = 150
    high_thresh = 200

    rho = 2
    theta = np.pi / 180
    thresh = 100
    min_line_len = 50
    max_line_gap = 150
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # 더 넓은 폭의 노란색 범위를 얻기위해 HSV를 이용한다.

    lower_yellow = np.array([20, 100, 100], dtype="uint8")
    upper_yellow = np.array([30, 255, 255], dtype="uint8")

    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray_image, 100, 255)

    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)  # 흰색과 노란색의 영역을 합친다.
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)  # Grayscale로 변환한 원본 이미지에서 흰색과 노란색만 추출

    gauss_gray = gaussian_blur(mask_yw_image, kernel_size)

    canny_edges = canny(gauss_gray, low_thresh, high_thresh)

    vertices = [get_pts(flag=0)]
    roi_image = region_of_interest(canny_edges, vertices)

    line_image = hough_lines(roi_image, rho, theta, thresh, min_line_len, max_line_gap)
    result = weighted_img(line_image, image, α=0.8, β=1., λ=0.)
    # cv2.polylines(result, vertices, True, (0, 255, 255)) # ROI mask

    return result, line_image


################################################################################################################추가

# 이미지 하단에서 검색된 이미지 사이에 거리 및 라인 그리기
def drowLine(bbox_xyxy, im0):
    for i in range(len(bbox_xyxy)):
        # 사진 기준
        # 왼쪽 상단
        xl = bbox_xyxy[i][0]
        # 왼쪽 하단
        yu = bbox_xyxy[i][1]
        # 오른쪽 상단
        xr = bbox_xyxy[i][2]
        # 오른쪽 하단
        yd = bbox_xyxy[i][3]

        # 거리 계산
        Distance = str(round((im0.shape[0] - yd) * (1 / 11.5), 2))  # 픽셀 : 실제 거리 = 1 : 1/11.5

        # 라인 그리기
        x2 = int(np.median(np.array([xl, xr])))
        # 15m 이하 거리면 빨간색 나머지 초록색 라인
        if float(Distance) < 15:
            cv2.line(im0, (x2, yd), (x2, im0.shape[0]), (0, 0, 255), 3)
        else:
            cv2.line(im0, (x2, yd), (x2, im0.shape[0]), (0, 255, 0), 3)

        font = cv2.FONT_HERSHEY_COMPLEX  # 폰트
        cv2.putText(im0, Distance + 'm', (xl, int(np.median(np.array([im0.shape[0], yd])))), font, 0.4, (0, 0, 255), 1,
                    cv2.LINE_AA)  # 거리값 보여주기


# 차선 인식
def laneDet(im0):
    stencil = np.zeros_like(im0[:, :, 0])

    point1 = [round(im0.shape[1] / 4), round(im0.shape[0] / 1.1)]
    point2 = [round(im0.shape[1] / 4), round(im0.shape[0] / 1.44)]
    point3 = [round(im0.shape[1] / 1.8), round(im0.shape[0] / 1.44)]
    point4 = [round(im0.shape[1] / 1.8), round(im0.shape[0] / 1.1)]
    polygon = np.array([[point1], [point2], [point3], [point4]])

    cv2.fillConvexPoly(stencil, polygon, 1)
    # apply frame mask
    masked = cv2.bitwise_and(im0[:, :, 0], im0[:, :, 0], mask=stencil)
    # apply image thresholding
    ret, thresh = cv2.threshold(masked, 130, 145, cv2.THRESH_BINARY)
    # apply Hough Line Transformation
    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 30, maxLineGap=200)

    # Plot detected lines
    try:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(im0, (x1, y1), (x2, y2), (255, 0, 0), 3)

    except TypeError:
        print('typeError!!!')


def detect(opt, save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = torch.load(weights, map_location=device)[
        'model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        view_img = True
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            ######################### 차선인지 ##################################

            prc_img, hough = process_image(im0)
            im0 = prc_img.copy()
            '''
            stencil = np.zeros_like(im0[:,:,0])
            polygon = np.array([[50,270], [220,160], [360,160], [480,270]])
            cv2.fillConvexPoly(stencil, polygon, 1)
            # apply frame mask
            masked = cv2.bitwise_and(im0[:,:,0], im0[:,:,0], mask=stencil)
            # apply image thresholding
            ret, thresh = cv2.threshold(masked, 130, 145, cv2.THRESH_BINARY)
            # apply Hough Line Transformation
            lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 30, maxLineGap=200)
            im0 = im0.copy()
            '''

            # Plot detected lines
            try:
                # for line in lines:
                # x1, y1, x2, y2 = line[0]
                # cv2.line(im0, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.imwrite("./saveImage/" + str(frame_idx) + '.png', im0)
            except TypeError:
                cv2.imwrite("./saveImage/" + str(frame_idx) + '.png', im0)
            ############## 차선 인식 ~~~!!!  #########################################
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs = []

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # Pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, im0)

                # 이미지 list 에 저장
                ################### my_code#############################
                # 이미지 저장을 위한 로직
                # if frame_idx % 90 == 1 :
                #     # 원본 이미지 저장
                #     cv2.imwrite("./saveImage/" + str(frame_idx) + "_ano.png", im0)
                ################### my_code#############################

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    draw_boxes(im0, bbox_xyxy, identities)
                    ##################### my_code   #################################

                    # 전체 이미지 크기
                    # im0.shape

                    # 검색된 box 정보
                    # bbox_xyxy

                    # 검색된 box 번호
                    # print(identities[0])

                    # print(identities)
                    # print(bbox_xyxy)
                    # print(len(bbox_xyxy))
                    # print(im0.shape)

                    # 거리 및 중앙에서 부터 박스 까지 선
                    drowLine(bbox_xyxy, im0)

                # cv2.imwrite("./saveImages/" + str(frame_idx) + "_label.png", im0)
                # # 이미지 저장을 위한 로직
                # if frame_idx % 30 == 1 :
                #     # 처리된 이미지
                #     cv2.imwrite("./saveImages/" + str(frame_idx) + "_label.png" , im0)

                ##################### my_code   #################################

                # Write MOT compliant results to file
                if save_txt and len(outputs) != 0:
                    for j, output in enumerate(outputs):
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2]
                        bbox_h = output[3]
                        identity = output[-1]
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                                                           bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:

                # 영상 제거 하기
                # cv2.imshow(p, im0)

                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                print('saving img!')
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    print('saving video!')
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='yolov5/weights/yolov5s.pt', help='model.pt path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='/inferenceimages', help='source')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    # class 0 is person
    parser.add_argument('--classes', nargs='+', type=int,
                        default=[0, 1, 2], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort", type=str,
                        default="deep_sort_pytorch/configs/deep_sort.yaml")

    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    with torch.no_grad():
        detect(args)
