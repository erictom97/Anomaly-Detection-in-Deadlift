import mediapipe as mp
import cv2
import numpy as np
import os
import pandas as pd
# import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import sys
from sklearn.neighbors import KernelDensity
import warnings
from itertools import count
import pickle
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('GenerateSkelemotion'), '..')))
import GenerateSkelemotion.GenerateSkeleton as gs


autoencoder_model = tf.keras.models.load_model('autoencoder500epoch.keras')
encoder_model = tf.keras.models.load_model('encoder500epoch.keras')
with open('../Deadlift Stages/deadlift_bot_rf.pkl', 'rb') as f:
    model_rf = pickle.load(f)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
landmarks = ['class']
for val in range(1, 33+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

def get_args():
    parser = argparse.ArgumentParser(description="Predict deadlift video")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    return parser.parse_args()

#Now, input unknown images and sort as Good or Anomaly
def check_anomaly(img_path,out_vector_shape,kde):
    density_threshold = -240000 #Set this value based on the above exercise
    reconstruction_error_threshold = 0.032# Set this value based on the above exercise
    img  = Image.open(img_path)
    img = np.array(img.resize((256,256)))
    # plt.imshow(img)
    img = img / 255.
    img = img[np.newaxis, :,:,:]
    encoded_img = encoder_model.predict([[img]]) 
    encoded_img = [np.reshape(img, (out_vector_shape)) for img in encoded_img] 
    density = kde.score_samples(encoded_img)[0] 

    reconstruction = autoencoder_model.predict([[img]])
    reconstruction_error = autoencoder_model.evaluate([reconstruction],[[img]], batch_size = 1)[0]
    # print('reconstruction_error: ', reconstruction_error)
    # print('density: ', density)
    if density < density_threshold or reconstruction_error > reconstruction_error_threshold:
        return "Anomaly"
        
    else:
        return "Good Deadlift Form"

def main(video_path):

    frames = gs.main(video_path, is_folder=False)[0]

    # Image_folder = "../Data/Orientation/Data/deadlift/"
    train_folder = "../Data/Orientation/Data/train"
    valid_folder = "../Data/Orientation/Data/valid"
    anomaly_folder = "../Data/Orientation/anomaly"
    save_path = os.path.join(video_path.split('.')[0], os.path.split(video_path)[1].split('.')[0]+'.avi')
    datagen = ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow_from_directory(
        directory=train_folder,
        batch_size=4,
        class_mode="input",
        shuffle=True,
        seed=42
    )

    valid_generator = datagen.flow_from_directory(
        directory=valid_folder,
        # target_size=(45,100),
        batch_size=4,
        class_mode="input",
        shuffle=True,
        seed=42
    )

    anomaly_generator = datagen.flow_from_directory(
        directory=anomaly_folder,
        # target_size=(45,100),
        batch_size=4,
        class_mode="input",
        shuffle=True,
        seed=42
    )

    #Get encoded output of input images = Latent space
    encoded_images = encoder_model.predict_generator(train_generator)

    # Flatten the encoder output because KDE from sklearn takes 1D vectors as input
    encoder_output_shape = encoder_model.output_shape #Here, we have 16x16x16
    out_vector_shape = encoder_output_shape[1]*encoder_output_shape[2]*encoder_output_shape[3]

    encoded_images_vector = [np.reshape(img, (out_vector_shape)) for img in encoded_images]

    #Fit KDE to the image latent data
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(encoded_images_vector)
    
    image_path = os.path.join(video_path.split('.')[0], 'images')
    labels = []
    file_dir = sorted(os.listdir(image_path))
    for file in file_dir:
        if file.endswith(".png"):
            # print(file)
            labels.append(check_anomaly(os.path.join(image_path,file),out_vector_shape,kde))

    cnt = count(0)
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    current_stage = ''
    # out = cv2.VideoWriter('test1.mp4')
    # Define the codec and create a VideoWriter object

    writer = cv2.VideoWriter(save_path, 
                            cv2.VideoWriter_fourcc(*'MJPG'), 30, size)
    # writer = cv2.VideoWriter(output_file, fourcc, fps, frame_size)
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
                        # print(body_language_class, body_language_prob)

                        if body_language_class == 'up' and body_language_prob[body_language_prob.argmax()] > 0.7:
                            current_stage = 'up'
                        elif body_language_class == 'down' and body_language_prob[body_language_prob.argmax()] > 0.7:
                            current_stage = 'down'
                        else:
                            current_stage = 'neutral'

                        # print(current_stage)

                        # cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
                        cv2.rectangle(image, (0,0), (280,220), (245,117,16), -1)

                        
                        # cv2.putText(image, 'CLASS',
                        #             (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                        # cv2.putText(image, current_stage,
                        #             (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                        cv2.putText(image, f'Stage: {current_stage.upper()}',
                                (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
                        cv2.putText(image, f'Probability: {str(round(body_language_prob[body_language_prob.argmax()],2))}',
                                (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
                        i = next(cnt)
                        cv2.putText(image, f'Frame : {str(i)}',
                                (10,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
                        
                        # cv2.putText(image, 'PROB',
                        #             (65,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                        # cv2.putText(image, str(round(body_language_prob[body_language_prob.argmax()],2)),
                        #             (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                        
                        # i = next(cnt)
                        # cv2.putText(image, str(i),
                        #             (10,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)
                        
                        for rep, (fr, lbl) in enumerate(zip(frames, labels),start=1):
                            # check i in between fr[0] and fr[1]
                            if i >= fr[0] and i <= fr[1]:
                                # print(i,lbl)
                                if lbl == 'Anomaly':
                                    cv2.rectangle(image, (600,0), (1000,70), (0,0,255), -1)

                                elif lbl == 'Good Deadlift Form':
                                    cv2.rectangle(image, (600,0), (1000,70), (0,255,0), -1)

                                cv2.putText(image, lbl,
                                    (620,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
                                cv2.putText(image, f'Rep: {str(rep)}',
                                    (10,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
                                break
                        k = cv2.waitKey(1)
                        cv2.imshow('Deadlift Feed', image)
                        writer.write(image)
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break
                except Exception as e:
                    print(e)
                    # print(results.pose_landmarks)
                    pass

            else:
                break

    cap.release()
    cv2.destroyAllWindows()
                
if __name__ == "__main__":
    args = get_args()
    main(args.video)

