import os
import pickle
import numpy as np
import cv2
import face_recognition
import cvzone
from pymongo import MongoClient
import gridfs
from datetime import datetime
import time  # Importing time module to manage display duration

# MongoDB initialization
client = MongoClient("mongodb://localhost:27017/")
db = client['face_attendance']
students_collection = db['students']
fs = gridfs.GridFS(db)

# OpenCV setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
imgBackground = cv2.imread('Resources/background.png')

# Import mode images
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = [cv2.imread(os.path.join(folderModePath, path)) for path in modePathList]

#Load encodings (if available)
try:
    with open('EncodeFile.p', 'rb') as file:
        encodeListKnown, studentIds = pickle.load(file)
    print("Encode File Loaded")
except FileNotFoundError:
    print("Encode file not found. Starting with empty encodings.")
    encodeListKnown = []
    studentIds = []

modeType = 0
counter = 0
id = -1
imgStudent = []

# Variables for display timeout
last_recognition_time = 0
display_duration = 3  # Time to display student info (in seconds)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = cv2.resize(imgModeList[modeType], (414, 633))

    modeType = 0
    studentInfo = None
    imgStudent = []

    if faceCurFrame:
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            if encodeListKnown:
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

                if len(faceDis) > 0:
                    matchIndex = np.argmin(faceDis)
                    if matches[matchIndex]:  # Existing student
                        id = studentIds[matchIndex]
                        studentInfo = students_collection.find_one({"student_id": id})

                        if studentInfo:
                            modeType = 1
                            y1, x2, y2, x1 = [val * 4 for val in faceLoc]
                            bbox = (55 + x1, 162 + y1, x2 - x1, y2 - y1)
                            imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)

                            # Retrieve student image if available
                            image_file = fs.find_one({"filename": f"{id}.png"})
                            if image_file:
                                imgStudent = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

                            last_recognition_time = time.time()  # Store time when student info was displayed

                            # Get today's date
                            today_date = datetime.now().strftime("%Y-%m-%d")

                            # Check if today's date is already in attendance_dates
                            if today_date not in studentInfo['attendance_dates']:
                                # Mark attendance and update the student's attendance data
                                students_collection.update_one(
                                    {"student_id": id},
                                    {
                                        "$inc": {"total_attendance": 1},
                                        "$push": {"attendance_dates": today_date},  # Add today to the attendance_dates list
                                        "$set": {"last_attendance_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                                    }
                                )
                                print(f"Attendance for {studentInfo['name']} marked for {today_date}")
                            else:
                                # If attendance has already been marked today, do nothing
                                modeType = 3  # Display 'already marked' mode or similar
                                print(f"Attendance already marked for {studentInfo['name']} today.")
                    else:
                        print("Face not recognized. Treating as a new student.")
                        modeType = 4  # Switch to "new student" mode
                else:
                    print("Face not recognized. Treating as a new student.")
                    modeType = 4  # Switch to "new student" mode
            else:
                print("No known faces in the system.")
                modeType = 4  # Switch to "new student" mode

    if modeType == 4:  # New student mode
        # Collect details for the new student
        student_id = input("Enter Student ID: ")
        student_name = input("Enter Student Name: ")
        major = input("Enter Major: ")
        year = input("Enter Year: ")

        # Save the new student to the database
        new_student = {
            "student_id": student_id,
            "name": student_name,
            "major": major,
            "year": year,
            "total_attendance": 0,
            "standing": "B",
            "starting_year": datetime.now().year,
            "last_attendance_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attendance_dates": []
        }

        students_collection.insert_one(new_student)
        print(f"New student added: {student_name}, ID: {student_id}")

        # Store the new face encoding
        encodeListKnown.append(encodeFace)
        studentIds.append(student_id)

        # Save the encodings to the pickle file
        with open("EncodeFile.p", 'wb') as file:
            pickle.dump([encodeListKnown, studentIds], file)

        # Save the student's image in GridFS
        img_filename = f"{student_id}.png"
        cv2.imwrite(f"Images/{img_filename}", img)
        with open(f"Images/{img_filename}", "rb") as img_file:
            fs.put(img_file, filename=img_filename)

        print(f"Image of {student_name} saved and associated with ID: {student_id}.")
        modeType = 1  # Switch back to recognition mode

    # Display student information only if the recognition time is within 3 seconds
    if studentInfo and (time.time() - last_recognition_time) <= display_duration:
        cv2.putText(imgBackground, str(studentInfo['total_attendance']), (861, 125),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        cv2.putText(imgBackground, str(studentInfo['major']), (1006, 550),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(imgBackground, str(id), (1006, 493),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

        (w, h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
        offset = (414 - w) // 2
        cv2.putText(imgBackground, str(studentInfo['name']), (808 + offset, 445),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)
        # # Resize the student image to fit the target region (216x216)
        # imgStudent_resized = cv2.resize(imgStudent, (216, 216))
        #
        # # Place the resized student image on imgBackground
        # imgBackground[175:175 + 216, 989:989 + 216] = imgStudent_resized

    # Show the updated frame
    cv2.imshow("Face Attendance", imgBackground)
    cv2.waitKey(1)

