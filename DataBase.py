from pymongo import MongoClient
# Connect to MongoDB (ensure MongoDB is running on localhost:27017)
client = MongoClient("mongodb://localhost:27017/")
db = client['face_attendance']
students_collection = db['students']

# This script
