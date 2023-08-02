import sys
# sys.path.append('skeleton_images/')
# from skeleton_images import GenerateSkeletonImages
import mediapipe as mp
import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import csv
import pickle
import random
from itertools import count
import warnings
warnings.filterwarnings("ignore")


def main(path_to_folder):

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    landmarks = ['class']
    for val in range(1, 33+1):
        landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

    with open("../Deadlift Stages/deadlift_bot_rf.pkl", "rb") as f:
        model_rf = pickle.load(f)

    files = os.listdir(path_to_folder)
    print(files)
    for file in files:
        if file.endswith('.mp4'):
            cap = cv2.VideoCapture(os.path.join(path_to_folder, file))
            current_stage = ''
            output = pd.DataFrame()
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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
                        
                        try:
                            if results.pose_landmarks is not None:
                                row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark],dtype=object).flatten()
                                X = pd.DataFrame([row],columns=landmarks[1:])
                                body_language_class = model_rf.predict(X)[0]
                                body_language_prob = model_rf.predict_proba(X)[0]
                                if body_language_class == 'up' and body_language_prob[body_language_prob.argmax()] > 0.7:
                                    current_stage = 'up'
                                elif body_language_class == 'down' and body_language_prob[body_language_prob.argmax()] > 0.7:
                                    current_stage = 'down'
                                else:
                                    current_stage = 'none'
                                X['Class'] = current_stage
                                X['Prob'] = body_language_prob[body_language_prob.argmax()]
                                output = output.append(X)
                                
                        except Exception as e:
                            print(e)
                            print(results.pose_landmarks)
                            pass

                    else:
                        break

            cap.release()
            cv2.destroyAllWindows()

            output.index = range(len(output))
            output_copy = output.copy(deep=True)
                
            output['rep'] = (output['Class'] != output['Class'].shift(1)).cumsum()

            # Find the beginning and end of each rep based on the maximum probability
            output['beginning'] = output.groupby('rep')['x16'].idxmax()
            output['end'] = output.groupby('rep')['x16'].idxmax().shift(-1) - 1
            output.at[output.index[-1], 'end'] = output.index[-1]

            idx = []
            for value in output['beginning']:
                if value.is_integer():
                    if output['Class'][int(value)] == 'down':
                        idx.append(int(value))

            save_path = os.path.join(path_to_folder, file.split('.')[0])
            os.makedirs(save_path, exist_ok=True)        


            start = idx[0]
            frames = []
            counter = count(0)
            for id in idx[1:]:
                end = id
                name = next(counter)
                for row,col in output[start:end].iterrows():
                    body_id = random.randint(0, 1000000)
                    with open('{}/rep_{}.skeleton'.format(save_path, name), 'a') as file:
                        if row == start:
                            file.write(str(end-start)+'\n')
                        file.write("1"+ "\n")
                        file.write(str(body_id)+ "\n")
                        file.write("33"+ "\n")
                        for i in range(1,34):
                            column_x = 'x{}'.format(i)
                            column_y = 'y{}'.format(i)
                            column_z = 'z{}'.format(i)
                            file.write(f"{col[column_x]} {col[column_y]} {col[column_z]}"+ "\n")

                frames.append((start, end))
                start = end

            os.system(f'python GenerateSkeletonImages.py --data_path "{save_path}" --img_type 1 --temp_dist 10 --path_to_save "{save_path}"')
            image_path = os.path.join(save_path, 'images')
            os.makedirs(image_path, exist_ok=True)

            for file in os.listdir(save_path):
                if file.endswith(".npz"):
                    data = np.load(os.path.join(save_path, file))
                    lst = data.files
                    for item in lst:
                        data = (data[item]* 255).astype(np.uint8)
                        image = cv2.applyColorMap(data,cv2.COLORMAP_HSV)
                        cv2.imwrite(os.path.join(image_path, os.path.split(save_path)[1]+'_'+file.split('.')[0]+'.png'), image)


if __name__ == '__main__':
    path_to_folder = sys.argv[1]
    main(path_to_folder)

