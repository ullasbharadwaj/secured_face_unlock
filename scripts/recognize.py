"""
######################################################################
    MobiFace Unlock - A secure Face unlock mechanishm for Laptops

    File : recognize.py
    Purpose: Perform face recognition and act on screen locking
             and unlocking

    Author: Ullas Bharadwaj
######################################################################
"""
import face_recognition
import cv2
import numpy as np
import time
import os
import keyring
from sklearn import svm
import glob
import importlib.util

import utils
from store_credentials import set_password
from store_credentials import get_password
from send_email import send_intruder_alert

"""
Import TensorFlow libraries required for Object Detection.
Object Detection is used to overcome the scenarios when the
face recognition library fails to recognize the Face even
when valid user is using the device.
"""

pkg = importlib.util.find_spec('tflite_runtime')
use_TPU  = False
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate


"""
    Train a SVM Classifier to assist the face recognition
"""

def trainSVM(faceModel, known_face_encodings):
    print("Training SVM!")
    faceModel.fit(known_face_encodings)
    filename = '../models/FaceModel.pickle'
    utils.WritePickle(faceModel, filename)
    utils.WriteConfig(known_face_encodings)
    print("Training Finished!")

"""
    Check if re-training of SVM model is needed. If any new data is added,
    retrain the SVM model
"""
def ReTrainingNeeded(NumTrainImages):
    return utils.ReadConfig(NumTrainImages)


"""
    Perform Face Recognition, SVM classification and Object detection
    when needed to decide on the Screen Lock or Unlock
"""
def startImageProcessing():
    trueFacesLoc = "/home/surabhiullas/Desktop/HobbyProjects/Face_Recognition/trueData/"

    # Load the label map
    with open("../models/labelmap.txt", 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Have to do a weird fix for label map if using the COCO "starter model" from
    # https://www.tensorflow.org/lite/models/object_detection/overview
    # First label is '???', which has to be removed.
    if labels[0] == '???':
        del(labels[0])

    interpreter = Interpreter("../models/detect.tflite")

    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5


    # Get a reference to webcam #0 (the default one)
    for camera in glob.glob("/dev/video?"):
        video_capture = cv2.VideoCapture(camera)
        if video_capture.isOpened():
            break

    # Initialize some variables
    known_face_encodings = []
    known_face_names = []
    face_locations = []
    face_encodings = []
    face_names = []

    process_this_frame = True

    previous_fps = 0
    average_fps = 30 # default value

    frame_counter = 0
    presence_counter = 0
    absence_counter = 0
    lock_flag = False
    trueFacesFound = False

    NumTrainImages = sum(len(fs) for _,_,fs in os.walk(trueFacesLoc))

    # Check if Re-training is needed
    if ReTrainingNeeded(NumTrainImages) == False:
        trainData = np.zeros((1,128))
        for trueImage in glob.iglob(trueFacesLoc + '**/*', recursive=True):
            if "jpg" in trueImage or "png" in trueImage:
                imageName = os.path.join(trueFacesLoc,trueImage)
                image = face_recognition.load_image_file(imageName)
                encoding = face_recognition.face_encodings(image)[0]
                known_face_encodings.append(encoding)
                face_names.append(trueImage)
                encoding = encoding.reshape((1, encoding.shape[0]))
                trainData = np.concatenate((trainData, encoding), axis=0)

        trainData = trainData[1:,...]
        faceModel = svm.OneClassSVM(kernel="rbf", degree=4, nu=0.1)#, gamma=0.05)
        trainSVM(faceModel, trainData)

    else:   # No re-training is needed. Already trained and no update in the training Data. Hence load the pre-trained model
        modelName = '../models/FaceModel.pickle'
        faceModel = utils.ReadPickle(modelName)
        known_face_encodings = utils.ReadPickle("../models/trianedData.pickle")

    # Get the Tick Frequency to calculate FPS
    freq = cv2.getTickFrequency()

    false_face_counter = 0

    # Store and modify password as required to unlock the gnome screen
    set_password()

    pwd = (" ".join(get_password()))

    while True:
        face_label = [] # place holder for the face_labels

        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        trueFacesFound = False

        # Grab a single frame of video
        try:
            ret, frame = video_capture.read()
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        except:
            continue

        frame_counter += 1

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []

            if len(face_encodings) > 0:
                falseFacesFound = True #Initialyy assume false face is detected
            else:
                falseFacesFound = False #No face found, hence no falseFace

            for face_encoding in face_encodings:

                img = face_encoding.reshape((1,face_encoding.shape[0]))
                prediction = faceModel.predict(img)

                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                if any(matches) == True or prediction == 1:
                    falseFacesFound = False
                    absence_counter = 0
                    false_face_counter = 0
                    trueFacesFound = True
                    face_label.append("known")

                    if  lock_flag == True:
                        presence_counter += 1
                        if presence_counter > 2:
                            os.popen('gnome-screensaver-command --deactivate && sleep 15 && xdotool key --delay 50 ' + pwd)
                            presence_counter = 0
                            lock_flag = False
                else:
                    face_label.append("unknown")
            """
            If only a False Face is detected, immediately lock the screen.
            If both actual user and false face is detected, then do not lock screen.
            """

            if falseFacesFound == True:
                false_face_counter += 1
                if false_face_counter > 5:
                    os.popen('gnome-screensaver-command --lock')
                    lock_flag = True
                    cv2.imwrite("intruder.jpg",frame)
                    send_intruder_alert("intruder.jpg")
                    print("Intruder alert sent")
                    false_face_counter = 0


            if lock_flag == False and trueFacesFound == False:
                """
                    When Face is not detected, perform Object Detection to Check
                    if the user is really not in place. Because, hand/objects over the
                    face or covering the face partially will make face_recognition
                    library Return 0 faces. In those cases do not increment the
                    absence counter.

                    This technique will not cause any third person to access
                    because, as soon as intruder showcases his wife, the screen
                    will be locked
                """
                lock_waiting_factor = 0.5
                do_not_increment = False
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (width, height))
                input_data = np.expand_dims(frame_resized, axis=0)

                # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
                if floating_model:
                    input_data = (np.float32(input_data) - input_mean) / input_std

                # Perform the actual detection by running the model with the image as input
                interpreter.set_tensor(input_details[0]['index'],input_data)
                interpreter.invoke()
                classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
                scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
                # Loop over all detections and draw detection box if confidence is above minimum threshold
                for i in range(len(scores)):
                    if ((scores[i] > 0.5) and (scores[i] <= 1.0)):
                        if int(classes[i]) == 0: # Check if person is detected, if then do not increment absence counter
                            lock_waiting_factor = 10
                            # do_not_increment = True


                # if do_not_increment == False:
                absence_counter += 1
                if absence_counter > lock_waiting_factor * average_fps: #3s assuming 15fps
                    os.popen('gnome-screensaver-command --lock')
                    absence_counter = 0
                    lock_flag = True


            t2 = cv2.getTickCount()
            average_fps = freq / (t2 - t1)
            cv2.putText(frame,'FPS: {0:.2f}'.format(average_fps),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
            utils.displayResultImage(frame, face_locations, face_label)

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("**************************************************\n")
    print("************* MobiFace Unlock ********************\n")
    print("*********** Author: Ullas Bharadwaj **************\n")
    print("**************************************************\n")
    startImageProcessing()
