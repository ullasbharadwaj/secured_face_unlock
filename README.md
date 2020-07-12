# MobiFace Unlock - A secured face unlock mechanism for workstations using MobileNets + Face Recognition Library and SVM classifier.

Nowadays, there are many workstations such as laptops with the availability of web cameras but most of them do not come with the support of face unlock mechanisms. Hence, users are required to provide passwords or fingerprints to gain access into the desktop. To avoid the hassle of password typing and provide a straightaway access into the desktop, I have come up with the MobiFace Unlock.

Now, you can unleash the potential of the webcam on workstations for improved security and ease of working.

This application combines the beauty and advantage of many worlds. It uses the face-recognition library from https://pypi.org/project/face-recognition/ , Single Shot MultiBox Detector with MobileNet networks to perform object detection and a simple One Class SVM classifier.

## Features
1. Screen unlock when a valid user sits in front of the PC, no password is needed to be entered

2. Automatically locks the desktop when an unrecognized or an intruder starts using the PC

3. Automatically locks the desktop when there is no activity

4. Provides seemless working space, even when faces of valid users are not recognized, example turning around, covering face with cellphone/objects etc.

5. Automatically sends an E-mail attached with the photo of the intruder and the location of the workstation (long and latitude) to a pre-defined Mail ID

6. Automatically retrains the syste,. when a new data sample is added to the trueData

## Potential Applications
1. Normal working days for hassle free face unlock feature
2. When you want to leave the laptop at your desk with some confidential data for a coffee break :-p You will have the intruder caught along with photographs

## Dependencies

OpenCV

Face Recognition Library

TensorFlow Lite Interpreter

Scikit-Learn

SMTP

## Procedure to deploy the application

The dependecies needs to be installed before hand.

The user needs to take some shots of the users to be recognitzed and place it in the trueData folder. The folder structure needs to be as follows.

trueData -> Multple Users 

Each user directory needs to have atleast 5 images for good results.

Once the above data is prepared which takes less than a minute as the webcam itself can be used to take pictures, follow below steps.

$: cd scripts

$: python3 recognize.py

The application prompts for the password of the desktop when started. This is to provide the unlock feature.

Currently the SMTP is set to the localhost. This means, I am directing the mails to the local SMTP server. In order to start the SMTP server, use below command in another terminal. However, you cannot see the images in this mode.

$: sudo python3 -m smtpd -c DebuggingServer -n localhost:25

In order to use the actual E-mail ID to receive the intruder alerts, enable Lines 51, 52 and 53 with the details of the mail addresses in send_email.py

You can launch the Python Script at the startup to have it running automatically.

Referenecs:
1. https://pypi.org/project/face-recognition/
2. https://docs.python.org/3/library/email.examples.html
