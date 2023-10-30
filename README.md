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

Here, you can provide detailed instructions on how to use your code to perform anomaly detection on action videos. Include code snippets, explanations, and usage examples to help users get started quickly.

```bash
python detect_anomalies.py --video_path video.mp4
```

## Data

In this section, explain how to obtain or prepare the dataset for your project. Include any necessary download links, instructions for data preprocessing, and organization.

## Model Training

Detail the process of training the autoencoder model, including hyperparameters, training data, and evaluation metrics.

## Results

Share the results and insights gained from your anomaly detection system. Visualize anomalies and provide a clear interpretation of the findings.

## Contributing

If you'd like to contribute to this project, please follow the guidelines in [CONTRIBUTING.md](CONTRIBUTING.md).

## References

[1] https://github.com/carloscaetano/skeleton-images
---

Customize the above template with your specific project details, and be sure to include relevant code snippets, documentation, and visuals to make your GitHub README informative and engaging for potential users and contributors.
 
