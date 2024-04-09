import cv2
import constants
from utils import calculate_embedding,StudentDataHandler
import torch # type: ignore
from torchvision.transforms import Resize, ToTensor # type: ignore
import json
import os
from facenet_pytorch import InceptionResnetV1 # type: ignore
from tqdm import tqdm as loading_bar
from os.path import exists

model = InceptionResnetV1(pretrained='vggface2').eval().to(constants.device)

data_dir='dataset'
class_names=os.listdir(data_dir)

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


def add_data_to_json(rollNo,name):
    with open('./static/data/students.json') as file:
        data = json.load(file)
    new_student = {
        "RollNo": rollNo,
        "Name": name,
        "image_path": "./dataset/"+name
    }
    data["students"][rollNo]=new_student
    sorted_data = sorted(data['students'].values(), key=lambda x: x['Name'])
    sorted_students = {student['RollNo']: student for student in sorted_data}
    data['students'] = sorted_students
    with open('./static/data/students.json', 'w') as file:
        json.dump(data, file, indent=4)
    print(data)

def new_register(new_class):
    tensor_file=constants.tensor_file_name
    tensor_list=[]
    if(exists(tensor_file)):
        tensor_list=torch.load(tensor_file)
    new_embedding=find_embedding(new_class)
    tensor_list.append(new_embedding)
    print(len(tensor_list))
    torch.save(tensor_list,tensor_file)
    
def find_embedding(new_class):
    class_dir = os.path.join(data_dir, new_class)
    class_embeddings = []
    for filename in loading_bar(os.listdir(class_dir)):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(class_dir, filename)
            image = cv2.imread(image_path)
            embedding = calculate_embedding(image,model)
            class_embeddings.append(embedding)
    return mean(class_embeddings)
    # class_embeddings_dict[class_name] = class_average_embedding
    

# Iterate over each class (person) in the dataset
# for class_name in loading_bar(class_names, desc='Processing dataset'):
#     class_dir = os.path.join(data_dir, class_name)
#     class_embeddings = []
#     # Iterate over each image in the class directory
#     for filename in loading_bar(os.listdir(class_dir)):
#         if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
#             image_path = os.path.join(class_dir, filename)
#             image = cv2.imread(image_path)
#             embedding = calculate_embedding(image,model)
#             class_embeddings.append(embedding)
#     # Calculate the average embedding for the class
#     class_average_embedding = mean(class_embeddings)
#     # class_embeddings_dict[class_name] = class_average_embedding
#     known_faces.append(class_average_embedding)
# print(class_embeddings_dict)




# Save the list of tensors to a file using torch.save()
