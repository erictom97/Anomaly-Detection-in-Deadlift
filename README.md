# Anomaly Detection in Deadlift

This repository contains the code and resources for the project that explores the use of Google's MediaPipe and OpenCV to detect skeleton keypoints in action videos, generating a skeleton image representation (Skelemotion [1]). The generated Skelemotion data is then utilized to train an autoencoder model for learning the movement patterns by minimizing the reconstruction error. Ultimately, the project is designed to detect anomalies in each repetition of the deadlift exercise.

![test1-4](https://github.com/erictom97/Anomaly-Detection-in-Deadlift/assets/40288848/2114b5a0-6bcd-4e01-aafe-26845e68d4d2)



## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Data](#data)
- [Model Training](#model-training)
- [Results](#results)
- [Contributing](#contributing)
- [References](#References)

## Introduction

The goal of this project is to create a robust anomaly detection system for deadlift exercises using a combination of computer vision techniques and deep learning. By capturing and analyzing the skeletal movement patterns, we aim to automatically identify anomalies in the execution of the exercise, which can be useful for fitness tracking, injury prevention, and form correction.

## Key Features

- Utilizes Google's MediaPipe for skeleton keypoint detection.
- Generates Skelemotion images for representing skeletal movements.
- Trains an autoencoder model to learn and reconstruct the movements.
- Implements anomaly detection algorithms to identify deviations from the learned patterns.
- Provides a clear visualization of anomalies in action videos.

## Getting Started

To get started with the project, follow these steps:

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/deadlift-anomaly-detection.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download or prepare your action video dataset (refer to the [Data](#data) section for details).

4. Follow the instructions in the [Usage](#usage) section to run the code and start the anomaly detection process.

## Usage

1. Navigate to the predict/predict.py directory.
- Use your file explorer or terminal to go to the directory where predict.py is located.
2. Open a terminal within this directory.
- You can open a terminal or command prompt in this directory by right-clicking and selecting
"Open Terminal" (or "Open Command Prompt" on Windows).
3. Activate the virtual environment where the necessary packages are installed.
- If you're using a virtual environment (e.g., conda or venv), activate it using the appropriate
command.
4. Execute the prediction script.
- Run the following command in the terminal:
```bash
python predict.py --video <path_to_deadlift_video>
```
Replace `<path_to_deadlift_video>` with the actual file path to your Deadlift video.
5. Wait for the result window to open.
- The prediction process may take some time depending on the video's length and complexity.
Please be patient.
6. Access your video results.
- Once the prediction is complete, the video results will be automatically saved in the same
directory as your input video file.

## Data

The following pipeline was followed to create the dataset for training:

<img width="921" alt="Screenshot 2023-10-30 at 7 35 34 pm" src="https://github.com/erictom97/Anomaly-Detection-in-Deadlift/assets/40288848/88fd16c4-a024-4964-9658-b6c78064ef3b">

Refer to the report document for a detailed explanation of each stage.


## Model Training

The skeleton representation image is fed to an autoencoder to reconstruct the dataset over 500 epochs.

<img width="921" alt="Screenshot 2023-10-30 at 7 38 12 pm" src="https://github.com/erictom97/Anomaly-Detection-in-Deadlift/assets/40288848/a0488a42-6b1c-4869-8ad2-a29ca276886c">


## References

[1] https://github.com/carloscaetano/skeleton-images
---

Customize the above template with your specific project details, and be sure to include relevant code snippets, documentation, and visuals to make your GitHub README informative and engaging for potential users and contributors.
 
