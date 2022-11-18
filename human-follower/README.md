# Human-Following-Robot

## Project Overview
The project uses a Machine Learning model - MobileNet SSD v1 (COCO), Python OpenCV and TensorFlow Lite interpreter to track a human object.
It performs human following and streams the robot view over LAN using FLASK (Web Framework).

## Human Following
The camera picks up frames which are then analysed by the ML Algorithm to calculate the position of any human in the frame. Based on the position in frame, a delay time value in calculated which instructs the motors to move for that amount of time in the calculated direction.
