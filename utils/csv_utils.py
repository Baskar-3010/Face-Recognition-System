import pandas as pd
import json
import datetime
import time
from datetime import date
from os.path import exists

class StudentDataHandler:
    def __init__(self, json_file, csv_file):
        self.json_file = json_file
        self.csv_file = csv_file+"_"+time.strftime("%H_%M_%S_", time.localtime())+str(date.today())+".csv"
        self.json_data = self.load_json_data()
        self.status = "Status"
        self.inTime="InTime"
        
        if exists(csv_file):
            self.csv_data = self.load_csv_data()
        else:
            self.create_csv_file()

    def get_all_regno(self):
        return [self.json_data["students"][regno]["RollNo"] for regno in self.json_data["students"]]

    def get_all_imagenames(self):
        return [self.json_data["students"][regno]["image_path"] for regno in self.json_data["students"]]
    
    def load_json_data(self):
        with open(self.json_file, 'r') as f:
            return json.load(f)

    def load_csv_data(self):
        return pd.read_csv(self.csv_file)
    
    def create_csv_file(self):
        # Create an empty DataFrame
        # empty_df = pd.DataFrame(columns=['reg_no', 'name', self.status])
        sample_data = {"RollNo": [],"Name": [], self.status : [], self.inTime : []}
        
        for reg_no in self.json_data["students"]:
            sample_data["RollNo"].append(reg_no)
            sample_data["Name"].append(self.json_data["students"][reg_no]["Name"])
            sample_data[self.status].append("A")
            sample_data[self.inTime].append("null")
        
        empty_df = pd.DataFrame(sample_data)
        
        # Save the empty DataFrame to a new CSV file
        empty_df.to_csv(self.csv_file, index=False,)
        # Load the newly created CSV file into self.csv_data
        self.csv_data = self.load_csv_data()
    
    def get_info_from_reg_no(self, reg_no):
        # Filter rows where 'reg_number' matches the given registration number
        info = self.csv_data[self.csv_data['RollNo'] == str(reg_no)]
        # print(self.csv_data['reg_no'] == int(reg_no))
        # Check if any row matches the registration number
        if not info.empty:
            return info.to_dict(orient='records')[0]
        else:
            return None
        
    def update_status_for_reg_no(self, reg_no, status, inTime):
        # Check if the given registration number exists in the DataFrame
        if reg_no in self.csv_data['RollNo'].values:
            # Update the status for the corresponding registration number and current date column
        
            self.csv_data.loc[self.csv_data['RollNo'] == reg_no, self.status] = status
            self.csv_data.loc[self.csv_data['RollNo'] == reg_no, self.inTime] = inTime
            # Save the updated DataFrame to the CSV file
            self.csv_data.to_csv(self.csv_file, index=False)
            print("Status updated successfully for registration number", reg_no)
        else:
            print("Registration number", reg_no, "not found.")