import torch
import cv2

# --------------------- inputs ----------------------

# known face  
database_path = "./known_images"
db_json_file = database_path + "/students.json"

#inference
"""
0 - webcam
'video.mp4' - input video
"""
input_format = 0
#input_format = "C:\\Users\\ybask\\OneDrive\\Desktop\\nitheesh.jpg"


students_json_file = "./static/data/students.json"
attendance_file = "./student_db/attendance/attendance"
known_images_folder = "./Color"  # Path to the folder of known images
tensor_file_name='tensor_list.pt'

threshold = 0.7

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


red = (255,0,0)
green = (0,255,0)
font = cv2.FONT_HERSHEY_SIMPLEX