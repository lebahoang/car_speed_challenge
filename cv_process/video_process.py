import cv2

def extract_frames(video_path='data/train.mp4', saving_folder='data/frames'):
    vc = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    fps = vc.get(cv2.CAP_PROP_FPS)
    print("Frame per second of this video: {}".format(fps))
    i = 1
    while True:
        next, frame = vc.read()
        if not next:
            break
        cv2.imwrite(saving_folder + '/{:d}.jpg'.format(i), frame)
        i += 1
