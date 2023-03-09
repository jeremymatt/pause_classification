import pandas as pd
import numpy as np
from Patient import Patient
    
class Patients:
    def __init__(self, csv_filename):
        self.p_list = []
        p_df = pd.read_csv(csv_filename)
         #print(p_df.columns.values[0])
        self.headers = p_df.columns.values
        #print(headers)
        #p_df.values[0][0]
        for p in p_df.values:
            #print(p[0])
            md = dict()
            for i in range(len(self.headers)):
                md[self.headers[i]] = p[i]
            patient = Patient(md.get("p0_patient_id"), md)
            self.p_list.append(patient)
        x = 19999
