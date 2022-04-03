# pyinstaller main.py --name AILabel --onefile --hidden-import Shapely --hidden-import yaml --hidden-import="PIL.E
# xifTags" --hidden-import seaborn
import os
import cv2
import torch
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
torch.cuda.empty_cache()
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best6.pt', force_reload=True)
CLASSNUMBER = 0
LABELS = []
CLASSES = []

# Get classes from Classes.txt file
with open('Classes.txt') as f:
    lines = f.readlines()
    for line in lines:
        CLASSES.append(line.rstrip())


# Updates the slider and text at the top right of the screen
def updateclass(input):
    global CLASSNUMBER
    global LABELS
    CLASSNUMBER = input
    drawImage(LABELS)
    return


downcoords = None
upcoords = None


def mousefunction(event, x, y, flags, param):
    if event == cv2.EVENT_RBUTTONDOWN:
        global LABELS
        Labels = LABELS
        image = cv2.imread(CURRENTIMAGE)
        NewLabels = []

        # Checking if the location right clicked was in or near one of the bounded boxes
        # If so, remove the label and redraw the boxes
        for label in Labels:
            point = Point(x, y)
            slabel = label.split(" ")
            height, width, channels = image.shape
            x_center, y_center, w, h = float(slabel[1]) * width, float(slabel[2]) * height, float(
                slabel[3]) * width, float(slabel[4]) * height
            x1 = round(x_center - w / 2)
            y1 = round(y_center - h / 2)
            x2 = round(x_center + w / 2)
            y2 = round(y_center + h / 2)

            polygon = Polygon([(x1, y2), (x2, y2), (x2, y1), (x1, y1)])
            if not polygon.buffer(20.0).contains(point):
                NewLabels.append(label)
            else:
                print("Removing Label")

        LABELS = NewLabels  # Update Labels List
        drawImage(NewLabels)  # ReDraw Image with new labels
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        global downcoords
        downcoords = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        global upcoords
        upcoords = [x, y]
        Labels = LABELS
        NewLabels = []
        image = cv2.imread(CURRENTIMAGE)
        inputX = image.shape[1]
        inputY = image.shape[0]
        x1 = downcoords[0]
        y1 = downcoords[1]
        x2 = upcoords[0]
        y2 = upcoords[1]
        width = (abs(x2 - x1))
        height = (abs(y2 - y1))
        yoloX = "{:.6f}".format(((x1 + x2) / 2) / inputX)
        yoloY = "{:.6f}".format(((y1 + y2) / 2) / inputY)
        yoloWidth = "{:.6f}".format(width / inputX)
        yoloHeight = "{:.6f}".format(height / inputY)
        global CLASSNUMBER

        label = str(CLASSNUMBER) + " " + yoloX + " " + yoloY + " " + yoloWidth + " " + yoloHeight
        Labels.append(label)
        LABELS = Labels  # Update labels list with added label

        drawImage(Labels)
        return


def drawImage(labels):
    global CURRENTIMAGE
    image = cv2.imread(CURRENTIMAGE)
    for label in labels:
        label = label.split(" ")
        height, width, channels = image.shape
        x_center, y_center, w, h = float(label[1])*width, float(label[2])*height, float(label[3])*width, float(label[4])*height
        x1 = round(x_center-w/2)
        y1 = round(y_center-h/2)
        x2 = round(x_center+w/2)
        y2 = round(y_center+h/2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)

        cv2.putText(
            img=image,
            text=CLASSES[int(label[0])],
            org=(x1, y1 - 10),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1.0,
            color=(0, 0, 0),
            thickness=3
        )

        cv2.putText(
            img=image,
            text=CLASSES[int(label[0])],
            org=(x1, y1 - 10),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1.0,
            color=(255, 255, 255),
            thickness=2
        )

    global CLASSNUMBER
    cv2.rectangle(image, (0, 0), (400, 40), (255, 255, 255), -1)
    im = cv2.putText(
        img=image,
        text=CLASSES[CLASSNUMBER],
        org=(5, 25),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=1.0,
        color=(0, 0, 0),
        thickness=2
    )

    cv2.imshow("AI Label", image)


def AIDetections(filename):
    directory = 'images'
    im = cv2.imread(filename)
    inputX = im.shape[1]
    inputY = im.shape[0]
    Labels = []
    results = model(filename)
    detections = results.xyxy[0]
    for detection in detections:
        x1 = int(detection[0].item())
        y1 = int(detection[1].item())
        x2 = int(detection[2].item())
        y2 = int(detection[3].item())
        width = (abs(x2 - x1))
        height = (abs(y2 - y1))
        yoloX = "{:.6f}".format(((x1 + x2) / 2) / inputX)
        yoloY = "{:.6f}".format(((y1 + y2) / 2) / inputY)
        yoloWidth = "{:.6f}".format(width / inputX)
        yoloHeight = "{:.6f}".format(height / inputY)
        classNum = int(detection[5].item())

        label = str(classNum) + " " + yoloX + " " + yoloY + " " + yoloWidth + " " + yoloHeight
        Labels.append(label)

    return Labels, im



CURRENTIMAGE = ""
def main():
    # CHANGE THIS TO THE DIRECTORY OF YOUR PHOTOS
    # FileList = []
    vidname = "vid11" #Enter whatever you want here, just so you don't overwrite past stuff
    cap = cv2.VideoCapture('/media/connor/14TBWD/DEEPLEARNINGPROJECT/DATA/NewData/Police/(Extraction_1.1)_AXON_Fleet_2_Video_2022-02-18_1615-2(1).mp4') # put video path here
    STARTFRAME = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, STARTFRAME) #Use this to fast forward to a frame

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video  file")

    # Check whether the specified path exists or not
    isExist = os.path.exists('images')
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs('images')
        print("Images Directory Created")

    cv2.namedWindow("AI Label", cv2.WINDOW_GUI_NORMAL)
    cv2.createTrackbar('r', 'AI Label', 0, len(CLASSES)-1, updateclass)

    framecounter = STARTFRAME
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            framecounter += 1
            if framecounter % 300 == 0: #change this to how many frames you want to skip
                cv2.imwrite(f"labels/{vidname}%d.jpg" % framecounter, frame)
                global CURRENTIMAGE
                CURRENTIMAGE = f"labels/{vidname}%d.jpg" % framecounter
                Labels, im = AIDetections(CURRENTIMAGE)
                drawImage(Labels)
                global LABELS
                LABELS = Labels
                cv2.setMouseCallback('AI Label', mousefunction, param=[LABELS])

                k = cv2.waitKey(0)
                if k == 49:  # 1 to Apply

                    print("APPLY")
                    # filename = image
                    Labels = LABELS

                    path = 'labels'  # SAVES LABELS TO THE LABELS FOLDER
                    # Check whether the specified path exists or not
                    isExist = os.path.exists(path)
                    if not isExist:
                        # Create a new directory because it does not exist
                        os.makedirs(path)
                        print("Labels Directory Created")

                    print("\nOUTPUT TO " + f'{vidname}{framecounter}.txt')
                    with open(os.path.join('labels/', f'{vidname}{framecounter}.txt'), 'w+') as f:
                        for label in Labels:
                            f.write(label + "\n")
                            print(label)
                        f.flush()
                        print('\n')

                    # os.rename(CURRENTIMAGE, os.path.join('labels', image))  # Move labled image to labels folder
                    continue

                elif k == ord('q'):  # Press q to exit
                    print("Exiting")
                    exit()


if __name__ == "__main__":
    main()
