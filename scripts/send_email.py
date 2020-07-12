"""
######################################################################
    MobiFace Unlock - A secure Face unlock mechanishm for Laptops

    File : send_email.py
    Purpose: Function to send an alert E-mail whenever an unknown
             person tries to access the system

    Author: Ullas Bharadwaj
######################################################################
"""
# Basic Imports
import os
import smtplib

# Imports required for sending email
import mimetypes
import email
import email.mime.application
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

#Get location of the laptop
import geocoder

def send_intruder_alert(imagePath):
    laptop_location = geocoder.ip('me')

    # Create a text/plain message
    intruder_alert = MIMEMultipart()
    intruder_alert['Subject'] = 'Intruder Alert'
    intruder_alert['From'] = 'sender@gmail.com'
    intruder_alert['To'] = 'receiver@gmail.com'

    ImgFileName = imagePath
    img_data = open(ImgFileName, 'rb').read()

    # Image attachment and also send the location of the laptop in terms of latitude and longitude
    text_msg = MIMEText("Alert ! An Intruder has access to your Laptop at the location " + str(laptop_location.latlng))
    intruder_alert.attach(text_msg)
    intruder_image = MIMEImage(img_data, name=os.path.basename(ImgFileName))
    intruder_alert.attach(intruder_image)


    smtp = smtplib.SMTP('localhost') # This line enables testing locally
    """
    Comment above line and use following statements instead to send via Gmail. This requires
    user to enable "Less secure App access" in the Gmail settings

    smtp = smtplib.SMTP('smtp.gmail.com')
    smtp.starttls()
    smtp.login('sender@gmail.com','password')
    """

    smtp.sendmail(intruder_alert['From'],[intruder_alert['From']], intruder_alert.as_string())
    smtp.quit()
