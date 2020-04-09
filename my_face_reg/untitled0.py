# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 01:54:34 2020

@author: HP
"""

import face_recognition
from sklearn import svm
import os
from PIL import Image
# Training the SVC classifier
# The training data would be all the face encodings from all the known images and the labels are their names
encodings = []
names = []

def trainning():
# Training directory
    train_dir = os.listdir('train_dir/')

    # Loop through each person in the training directory
    for person in train_dir:
        pix = os.listdir("train_dir/" + person)
        
        # Loop through each training image for the current person
        for person_img in pix:
            # Get the face encodings for the face in each image file
            face = face_recognition.load_image_file("train_dir/" + person + "/" + person_img)
            face_bounding_boxes = face_recognition.face_locations(face)
            
            #If training image contains exactly one face
            if len(face_bounding_boxes) == 1:
                face_enc = face_recognition.face_encodings(face)[0]
                # Add face encoding for current image with corresponding label (name) to the training data
                encodings.append(face_enc)
                names.append(person)
            else:
                print(person + "/" + person_img + " was skipped and can't be used for training")
                    
                    # Create and train the SVC classifier
    clf = svm.SVC(gamma='scale')
    clf.fit(encodings,names)
    return clf

def match(clf):
    print("Press 1 to provide an address for test Image and press 2 if the test_image is present in current working directory")
    opt = int(input())
    if(opt==1):
        test_image_add = input("Enter The address of test image")
        try:
            img = Image.open(test_image_add)
            img.save("test_image.jpg");
        except IOError:
            print("Wrong Addres Was provided using current directory as address for test image")
            test_image_add = os.curdir
            pass
    elif(opt==2):
        test_image_add = os.curdir
    else:
        print("Invalid input using current director as address")
        test_image_add = os.curdir
    
    # Load the test image with unknown faces into a numpy array
    test_image = face_recognition.load_image_file('test_image.jpg')
    # Find all the faces in the test image using the default HOG-based model
    face_locations = face_recognition.face_locations(test_image)
    no = len(face_locations)
    print("Number of faces detected: ", no)
    
    # Predict all the faces in the test image using the trained classifier
    print("Found:")
    for i in range(no):
        test_image_enc = face_recognition.face_encodings(test_image)[i]
        name = clf.predict([test_image_enc])
        print(*name)
while True:
    print("\n Menu\n(1) Train program on a dataset\n(2) Use the program to identify a person in the image\n(0) Exit the program")
    choice = int(input())
    if choice == 0:
        break
    elif choice==1:
        clf1 = trainning()
    elif choice ==2:
          match(clf1)
    else:
        print("Invalid choice plz choose again\n")
print("Bye")