# YoloV5-AI-Label
YoloV5 AI Assisted Labeling

![Example Image](https://i.imgur.com/62AzXtD.jpg)
YoloV5 model automatically labels, then you can make corrections before saving to a label file

1. Enter your classes in the Classes.txt file, starting with class 0
2. Replace the best.pt YoloV5 model with your own YoloV5 model
3. Place your unlabled images in the images folder
4. Run main.py to start the program.
5. Use the slider to select the class you want to label
6. Click and drag on the image to draw label boxes
7. Right click on a box to remove it
8. Once everything in the frame is labeled, press 1 to apply
9. Image and its label file will be moved to the labels folder

With these labels, you can train your model with [YoloV5](https://github.com/ultralytics). Once your new model is trained, replace the best.pt file with your new model
and start labeling again. As the model gets better, less manual labeling will be needed. 

###### Requirements
- OpenCV
- Shapely
- PyTorch
- [YoloV5 Requirements](https://github.com/ultralytics/yolov5/blob/master/requirements.txt)
