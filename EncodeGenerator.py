import cv2
import os
import pickle
import face_recognition
from pymongo import MongoClient
import gridfs

client = MongoClient("mongodb://localhost:27017/")
db = client['face_attendance']
fs = gridfs.GridFS(db)

# Import student images
folderPath = 'Images'
pathList = os.listdir(folderPath)
print("Image Paths:", pathList)

imgList, studentIds = [], []
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    student_id = os.path.splitext(path)[0]
    studentIds.append(student_id)

    # Upload image to GridFS if not already present
    if not fs.exists({"filename": f"{student_id}.png"}):
        with open(os.path.join(folderPath, path), "rb") as img_file:
            fs.put(img_file, filename=f"{student_id}.png")

print("Encoding Started ...")
encodeListKnown = [face_recognition.face_encodings(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[0] for img in imgList if face_recognition.face_encodings(img)]
encodeListKnownWithIds = [encodeListKnown, studentIds]

# Save encodings to a pickle file
with open("EncodeFile.p", 'wb') as file:
    pickle.dump(encodeListKnownWithIds, file)
print("Encoding File Saved")
