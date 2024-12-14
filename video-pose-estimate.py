import cv2
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt, strip_optimizer, xyxy2xywh
from utils.plots import output_to_keypoint, plot_skeleton_kpts, colors, plot_one_box_kpt
import csv
import os  # 파일 경로 처리를 위한 모듈 임포트

@torch.no_grad()
def run(poseweights="yolov7-w6-pose.pt", source="football1.mp4", device='cpu', view_img=False,
        save_conf=False, line_thickness=3, hide_labels=False, hide_conf=True):

    frame_count = 0  # count number of frames
    total_fps = 0  # count total fps
    time_list = []   # list to store time
    fps_list = []    # list to store fps
    
    device = select_device(opt.device)  # select device
    half = device.type != 'cpu'

    model = attempt_load(poseweights, map_location=device)  # Load model
    _ = model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
   
    if source.isnumeric():
        cap = cv2.VideoCapture(int(source))  # pass video to VideoCapture object
    else:
        cap = cv2.VideoCapture(source)  # pass video to VideoCapture object
   
    if not cap.isOpened():  # check if VideoCapture is not opened
        print('Error while trying to read video. Please check path again')
        raise SystemExit()

    frame_width = int(cap.get(3))  # get video frame width
    frame_height = int(cap.get(4))  # get video frame height

    output_dir = "output"
    if not os.path.exists(output_dir) :
        os.makedirs(output_dir)

    csv_output_dir = "output_csv"
    if not os.path.exists(csv_output_dir) :
        os.makedirs(csv_output_dir)
    

    vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0]  # init VideoWriter
    resize_height, resize_width = vid_write_image.shape[:2]
    out_video_name = f"{output_dir}/{source.split('/')[-1].split('.')[0]}.mp4"
    out = cv2.VideoWriter(out_video_name,
                          cv2.VideoWriter_fourcc(*'mp4v'), 30,
                          (resize_width, resize_height))

    # 비디오 소스 이름에서 확장자를 제거하고 CSV 파일 이름 생성
    base_name = os.path.splitext(os.path.basename(source))[0]  # 파일 이름에서 경로 및 확장자 제거
    csv_file_name = f'{csv_output_dir}/{base_name}.csv'  # CSV 파일 이름 생성

    # CSV 파일 열기 (기존 파일이 있으면 덮어쓰고, 없으면 새로 생성)
    with open(csv_file_name, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)  # CSV 작성자 객체 생성
        csv_writer.writerow(['Object Index', 'Body Part', 'X Coordinate', 'Y Coordinate', 'Confidence'])  # 헤더 작성

        while cap.isOpened():  # loop until cap opened or video not complete
            print("Frame {} Processing".format(frame_count + 1))

            ret, frame = cap.read()  # get frame and success from video capture
            
            if ret:  # if success is true, means frame exists
                orig_image = frame  # store frame
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)  # convert frame to RGB
                image = letterbox(image, (frame_width), stride=64, auto=True)[0]
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
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                for i, pose in enumerate(output_data):  # detections per image
                    if len(output_data):  # check if no pose
                        for c in pose[:, 5].unique():  # Print results
                            n = (pose[:, 5] == c).sum()  # detections per class
                            print("No of Objects in Current Frame : {}".format(n))
                        
                        for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:, :6])):  # loop over poses for drawing on frame
                            c = int(cls)  # integer class
                            kpts = pose[det_index, 6:]

                            # 키포인트 좌표 출력
                            print(f"Keypoints for detected object {det_index + 1}: {kpts.cpu().numpy()}")

                            # CSV에 신체 부위와 키포인트 저장
                            body_parts = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist',
                                          'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 'LEye', 'REar', 'LEar']
                            
                            for j in range(len(kpts) // 3):  # 각 키포인트마다 신체 부위, x, y 좌표와 신뢰도를 저장
                                if j < len(body_parts):  # body_parts 리스트의 길이를 초과하지 않도록
                                    x_coord = kpts[j * 3]  # x 좌표
                                    y_coord = kpts[j * 3 + 1]  # y 좌표
                                    confidence = kpts[j * 3 + 2]  # 신뢰도
                                    csv_writer.writerow([det_index + 1, body_parts[j], x_coord, y_coord, confidence])  # CSV에 쓰기
                            
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True), 
                                              line_thickness=line_thickness, kpt_label=True, kpts=kpts, steps=3, 
                                              orig_shape=im0.shape[:2])

                end_time = time.time()  # Calculation for FPS
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1
                
                fps_list.append(total_fps)  # append FPS in list
                time_list.append(end_time - start_time)  # append time in list
                
                # Stream results
                if view_img:
                    cv2.imshow("YOLOv7 Pose Estimation Demo", im0)
                    cv2.waitKey(1)  # 1 millisecond

                out.write(im0)  # writing the video frame

            else:
                break  # 프레임이 더 이상 없으면 루프 종료

        cap.release()
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")
        
        # plot the comparison graph
        plot_fps_time_comparision(time_list=time_list, fps_list=fps_list)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='football1.mp4', help='video/0 for webcam') #video source
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')   #device arugments
    parser.add_argument('--view-img', action='store_true', help='display results')  #display results
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels') #save confidence in txt writing
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)') #box linethickness
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels') #box hidelabel
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences') #boxhideconf
    opt = parser.parse_args()
    return opt

#function for plot fps and time comparision graph
def plot_fps_time_comparision(time_list,fps_list):
    plt.figure()
    plt.xlabel('Time (s)')
    plt.ylabel('FPS')
    plt.title('FPS and Time Comparision Graph')
    plt.plot(time_list, fps_list,'b',label="FPS & Time")
    plt.savefig("FPS_and_Time_Comparision_pose_estimate.png")
    

#main function
def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device,opt.poseweights)
    main(opt)
