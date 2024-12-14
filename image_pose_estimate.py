import cv2
import time
import torch
import argparse
import numpy as np
import os
import json  # JSON 모듈 추가
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt, strip_optimizer, xyxy2xywh
from utils.plots import output_to_keypoint, plot_one_box_kpt, colors

@torch.no_grad()
def run(poseweights="yolov7-w6-pose.pt", source="./data_image/plank.jpg", output_dir=".", device='cpu',
        line_thickness=3, hide_labels=False, hide_conf=True):

    device = select_device(device)  # select device
    half = device.type != 'cpu'

    model = attempt_load(poseweights, map_location=device)  # Load model
    _ = model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names

    # 이미지 읽기
    orig_image = cv2.imread(source)
    if orig_image is None:
        print('Error while trying to read image. Please check path again')
        raise SystemExit()

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)  # convert frame to RGB
    image = letterbox(image, (640, 640), stride=64, auto=True)[0]
    image_ = image.copy()
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    
    image = image.to(device)  # convert image data to device
    image = image.float()  # convert image to float precision (cpu)

    start_time = time.time()  # start time for fps calculation

    with torch.no_grad():  # get predictions
        output_data, _ = model(image)

    output_data = non_max_suppression_kpt(output_data,   # Apply non max suppression
                                            0.25,   # Conf. Threshold.
                                            0.65,  # IoU Threshold.
                                            nc=model.yaml['nc'],  # Number of classes.
                                            nkpt=model.yaml['nkpt'],  # Number of keypoints.
                                            kpt_label=True)

    output = output_to_keypoint(output_data)

    im0 = image[0].permute(1, 2, 0) * 255  # Change format [b, c, h, w] to [h, w, c] for displaying the image.
    im0 = im0.cpu().numpy().astype(np.uint8)
    
    im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)  # reshape image format to (BGR)

    keypoints_data = []  # 키포인트 데이터를 저장할 리스트
    body_parts = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 
                  'LShoulder', 'LElbow', 'LWrist', 'RHip', 'RKnee', 
                  'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 
                  'LEye', 'REar', 'LEar']  # 신체 부위 이름 리스트

    for i, pose in enumerate(output_data):  # detections per image
        if len(output_data):  # check if no pose
            for c in pose[:, 5].unique():  # Print results
                n = (pose[:, 5] == c).sum()  # detections per class
                print("No of Objects detected: {}".format(n))

            for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:, :6])):  # loop over poses for drawing on frame
                c = int(cls)  # integer class
                kpts = pose[det_index, 6:]

                # 키포인트 좌표 출력
                print(f"Keypoints for detected object {det_index + 1}: {kpts.cpu().numpy()}")

                # 키포인트 데이터 저장
                keypoints = []
                for j in range(len(kpts) // 3):  # 각 키포인트마다 신체 부위, x, y 좌표와 신뢰도를 저장
                    if j < len(body_parts):  # body_parts 리스트의 길이를 초과하지 않도록
                        x_coord = kpts[j * 3]  # x 좌표
                        y_coord = kpts[j * 3 + 1]  # y 좌표
                        confidence = kpts[j * 3 + 2]  # 신뢰도
                        
                        # 신체 부위, x, y 좌표와 신뢰도를 포함한 키포인트 저장
                        keypoints.append({
                            'body_part': body_parts[j],
                            'x': x_coord.item(),
                            'y': y_coord.item(),
                            'confidence': confidence.item()
                        })

                # 키포인트 데이터 추가
                keypoints_data.append({
                    'object_index': det_index + 1,
                    'keypoints': keypoints
                })

                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True), 
                                  line_thickness=line_thickness, kpt_label=True, kpts=kpts, steps=3, 
                                  orig_shape=im0.shape[:2])

    # 결과 이미지 이름 생성
    base_name = os.path.splitext(os.path.basename(source))[0]  # 입력 이미지 이름에서 확장자 제거
    
    # 현재 작업 디렉토리 가져오기
    current_directory = os.getcwd()
    
    # output 폴더 경로 설정
    output_dir = os.path.join(current_directory, "output")
    json_output_dir = os.path.join(current_directory, "output_json")

    # output 폴더가 존재하지 않으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(json_output_dir) :
        os.makedirs(json_output_dir)

    output_image_name = os.path.join(output_dir, f"{base_name}.jpg")  # 저장 경로 및 이름 설정
    cv2.imwrite(output_image_name, im0)  # 결과 이미지 저장
    print(f"Output image saved as {output_image_name}")

    # JSON 파일로 키포인트 저장
    keypoints_json_name = os.path.join(json_output_dir, f"{base_name}.json")  # JSON 파일 이름 설정
    with open(keypoints_json_name, 'w') as json_file:
        json.dump(keypoints_data, json_file, indent=4)  # JSON 파일에 저장
    print(f"Keypoints saved as {keypoints_json_name}")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='input_image.jpg', help='image path')  # 이미지 경로
    parser.add_argument('--output-dir', type=str, default='.', help='output directory')  # 결과 이미지 저장 경로
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')   # device arguments
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)') # box line thickness
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels') # box hide label
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences') # box hide confidence
    opt = parser.parse_args()
    return opt

# main function
def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device, opt.poseweights)  # strip_optimizer가 정의되어 있는지 확인
    main(opt)
