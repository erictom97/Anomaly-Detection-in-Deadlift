import mediapipe as mp
import cv2
import numpy as np
import csv
from matplotlib import pyplot as plt
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser(description="Annotate deadlift video")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--csv", type=str, required=True, help="Path to csv file")
    return parser.parse_args()

def export_landmark(results, action, path_to_csv):
    try:
        keypoints = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark],dtype=object).flatten()
        keypoints = np.insert(keypoints,0, action)
        with open(path_to_csv, 'a',newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(keypoints)
    except Exception as e:
        print(e)
        pass

def main(video_path, path_to_csv):

    """
        Annotate video and export landmarks to csv file
        Args:
            video_path: path to video file
            path_to_csv: path to csv file

        Call:
            python annotation.py --video <path_to_video> --csv <path_to_csv>
    """

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose 
    landmarks = ['class']
    for val in range(1, 33+1):
        landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

    if not os.path.exists(path_to_csv):
        with open(path_to_csv, 'w') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(landmarks)

    cap = cv2.VideoCapture(video_path)
    marks = []
    with mp_pose.Pose(min_detection_confidence=0.75, min_tracking_confidence=0.75) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                        )
                cv2.rectangle(image, (0,0), (280,200), (245,117,16), -1)
                cv2.putText(image, 'Press u: Up',
                                (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, 'Press d: Down',
                                (10,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, 'Press q: Quit',
                                (10,180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
                k = cv2.waitKey(1)
                if k == ord('u'):
                    export_landmark(results, 'up', path_to_csv)
                elif k == ord('d'):
                    export_landmark(results, 'down', path_to_csv)
                # else:
                #     export_landmark(results, 'none', path_to_csv)

                cv2.imshow('Annotation Feed', image)
                if k == ord('q'):
                    break
            else:
                print("End of video",ret)
                break

    cap.release()
    cv2.destroyAllWindows()

    
if __name__ == "__main__":
    args = get_args()
    main(args.video, args.csv)

    