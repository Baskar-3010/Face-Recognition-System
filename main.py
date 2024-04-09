import os
import cv2
import numpy as np 
import time
import json
from os.path import exists
from facenet_pytorch import InceptionResnetV1
import torch
from retinaface import RetinaFace

from torchvision.transforms import Resize, ToTensor
import torch.nn.functional as F  

from tqdm import tqdm as loading_bar
from datetime import date
import sys

import constants
from utils import calculate_embedding,StudentDataHandler


def main_fun():
    
    # Load the PyTorch FaceNet model
    model = InceptionResnetV1(pretrained='vggface2').eval().to(constants.device)
    
    # Load RetinaFace detector
    detector = RetinaFace(quality="normal")
    
    # creating object for student csv handler 
    student_db_handler = StudentDataHandler(json_file=constants.students_json_file,
                                            csv_file=constants.attendance_file)
    
    students_reg_no = student_db_handler.get_all_regno()
    
    # Load known faces and their corresponding names
    known_faces = []
    known_names = []
    data_dir='dataset'
    class_names=os.listdir(data_dir)
    class_embeddings_dict={}
    
    with open(constants.students_json_file,"r") as f:
        _json_data = json.load(f)
    known_names = [_json_data["students"][reg_no]["Name"] for reg_no in _json_data['students']]
    
    # known_names=[]
    
    # def extract_embedding(image):
    #     image_tensor = ToTensor()(image)
    #     with torch.no_grad():
    #         embedding = model(image_tensor.unsqueeze(0)).detach()
    #     return embedding
    # known_embedding=[]
    
    
    def mean(tensor_list):
        # tensor_list = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), torch.tensor([7, 8, 9])]
        # Initialize a tensor to store the mean
        mean_tensor = torch.zeros_like(tensor_list[0])
        # Compute the mean element-wise
        for tensor in tensor_list:
            mean_tensor += tensor
        mean_tensor /= len(tensor_list)
        print(mean_tensor)
        return mean_tensor
    
    if exists(constants.tensor_file_name):
        known_faces=torch.load(constants.tensor_file_name)
        # print(known_faces)
    else:
        # Iterate over each class (person) in the dataset
        for class_name in loading_bar(class_names, desc='Processing dataset'):
            class_dir = os.path.join(data_dir, class_name)
            class_embeddings = []
            # Iterate over each image in the class directory
            for filename in loading_bar(os.listdir(class_dir),desc=class_name):
                if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                    image_path = os.path.join(class_dir, filename)
                    image = cv2.imread(image_path)
                    embedding = calculate_embedding(image,model)
                    class_embeddings.append(embedding)
            # Calculate the average embedding for the class
            class_average_embedding = mean(class_embeddings)
            class_embeddings_dict[class_name] = class_average_embedding
            known_faces.append(class_average_embedding)
        torch.save(known_faces,constants.tensor_file_name)
        print(class_embeddings_dict)
    #initialize video capture
    video_capture = cv2.VideoCapture(constants.input_format)
    
    # Initialize variables for FPS calculation
    start_time = time.time()
    frame_count = 0
    
    while video_capture.isOpened():
        ret,frame = video_capture.read()
        if not ret or frame is None:
            print("Error: Failed to capture frame from webcam")
            break
        try:
            try:
                faces = detector.predict(frame)
            except Exception as e:
                print("Error occured at face prediction")
                
            for idx,face in enumerate(faces):
                if all(key in face for key in ['x1', 'y1', 'x2', 'y2']):
                    x1, y1, x2, y2 = face['x1'], face['y1'], face['x2'], face['y2']
                    # Extract face region from the frame
                    cropped_face = frame[y1:y2, x1:x2]
                    # Preprocess and extract embedding for the detected face
                    detected_embedding = calculate_embedding(cropped_face,model)
                    similarities = [F.cosine_similarity(detected_embedding, known_embedding).item() for known_embedding in known_faces]
                    max_index = np.argmax(similarities)
                    # print(known_faces[0])
                    # print(max_index)
                    if similarities[max_index] > constants.threshold:
                        name = known_names[max_index]
                        current_student_regno = students_reg_no[max_index]
                        info = dict(student_db_handler.get_info_from_reg_no(current_student_regno))
                        print(info)
                        col_name = "Status"
                    
                        if info[col_name] in ["P","A"]:
                            if info["Status"] == "A":
                                student_db_handler.update_status_for_reg_no(current_student_regno,"P",time.strftime("%H:%M:%S", time.localtime()))
                        else:
                            student_db_handler.update_status_for_reg_no(current_student_regno,"A","-")
                    else:
                        name = "unknown"
                
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    
                    
    
        except Exception as e:
            print("error: ",e)
    
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
    
        # Display FPS on the frame
        cv2.putText(frame, f"FPS: {round(fps, 2)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
        cv2.imshow("frame",frame)
        k = cv2.waitKey(1) 
        if k == 27:
            break
    video_capture.release()
    cv2.destroyAllWindows()


