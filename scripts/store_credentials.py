"""
######################################################################
    MobiFace Unlock - A secure Face unlock mechanishm for Laptops

    File : store_credentials.py
    Purpose: Store the screen lock screen password as a keyring 

    Author: Ullas Bharadwaj
######################################################################
"""

import keyring
from getpass import getpass

def set_password():
    print("Enter the screen unlock password now.")
    keyring.set_password('Master', 'xkcd', getpass())

def get_password():
    return keyring.get_password('Master', 'xkcd')
