import json
with open('./student_db/students.json', 'r') as file:
    data = json.load(file)
sorted_data = sorted(data['students'].values(), key=lambda x: x['Name'])
sorted_students = {student['RollNo']: student for student in sorted_data}
data['students'] = sorted_students
with open('./student_db/students.json', 'w') as file:
    json.dump(data, file, indent=4)
