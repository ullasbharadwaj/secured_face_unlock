"""
######################################################################
    MobiFace Unlock - A secure Face unlock mechanishm for Laptops

    File : utils.py
    Purpose: Some functions to assist main algorithm

    Author: Ullas Bharadwaj
######################################################################
"""

import pickle
import os
import pickle
import csv
import cv2
"""
Read function for pickle files
"""
def ReadPickle(filename):
        f = open(filename,"rb")
        data = pickle.load(f)
        f.close()
        return data

"""
Write function for pickle files
"""
def WritePickle(data, outfile):
        f = open(outfile, "w+b")
        pickle.dump(data, f)
        f.close()

"""
Write data to a configuration file
"""
def WriteConfig(known_face_encodings):
    with open('../models/config.pickle', 'w', newline='') as csvfile:
        fieldnames = ['Trained', 'SizeTrainData', "EncodingSize"]
        config = csv.DictWriter(csvfile, fieldnames=fieldnames)
        config.writeheader()
        config.writerow({'Trained': 'Yes', 'SizeTrainData': known_face_encodings.shape[0],\
                    'EncodingSize': known_face_encodings.shape[1]})
        WritePickle(known_face_encodings, "../models/trianedData.pickle")

"""
Read Data from a configuration file
"""
def ReadConfig(NumTrainImages):
    configFileName = "../models/config.pickle"
    modelFileName = "../models/FaceModel.pickle"
    if os.path.isfile(configFileName) and  os.path.isfile(modelFileName):
        with open(configFileName, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0
            for row in csv_reader:
                if int(row['SizeTrainData']) == int(NumTrainImages):
                    return True # No training required
                else:
                    return False # training required
    return False # training required

"""
    Function to display the face recognition output
"""
def displayResultImage(frame, face_locations, face_label):
    # Display the results
    cnt = 0
    for top, right, bottom, left in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, face_label[cnt], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cnt += 1

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return
