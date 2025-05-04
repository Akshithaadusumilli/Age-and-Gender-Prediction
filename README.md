# Age-and-Gender-Prediction

import cv2

# Load model files for face, age, and gender detection
faceProto = "C:/Users/akshi/Downloads/opencv_face_detector.pbtxt"
faceModel = "C:/Users/akshi/Downloads/opencv_face_detector_uint8.pb"

ageProto = "C:/Users/akshi/Downloads/age_deploy.prototxt"
ageModel = "C:/Users/akshi/Downloads/age_net (1).caffemodel"

genderProto = "C:/Users/akshi/Downloads/gender_deploy.prototxt"
genderModel = "C:/Users/akshi/Downloads/gender_net.caffemodel"

# Load networks
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Mean values used during model training (from Caffe)
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Age and gender label lists
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
           '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Open webcam
video = cv2.VideoCapture(0)
padding = 20

# Function to detect faces
def faceBox(net, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), 
                                 [104, 117, 123], swapRB=False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame, bboxes

# Main loop
while True:
    ret, frame = video.read()
    frame, bboxes = faceBox(faceNet, frame)

    for bbox in bboxes:
        face = frame[max(0, bbox[1]-padding):
                     min(bbox[3]+padding, frame.shape[0]-1),
                     max(0, bbox[0]-padding):
                     min(bbox[2]+padding, frame.shape[1]-1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict gender
        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]

        # Predict age
        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = ageList[agePred[0].argmax()]

        label = "{},{}".format(gender, age)
        cv2.putText(frame, label, (bbox[0], bbox[1]-10),
                    cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 2)

    # Display the output
    cv2.imshow("Age-Gender", frame)

    # Exit when 'q' is pressed
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()

