import csv
import os

def get_file_list(folder:str, ext:str='.csv'):
    return sorted([os.path.join(folder, f) for f in os.listdir(os.path.join(folder)) if f.endswith(ext)])



def save_csv(file_path, data, header=['starttime', 'endtime']):
    with open(file_path, 'w') as file:
        wr = csv.writer(file, delimiter=';', lineterminator='\n', quoting=0)
        wr.writerow(header)
        for call in data:
            wr.writerow(call)



def parse_metadata(filename: str):
    parts = os.path.basename(filename).split('.')[0].split('_')

    return {'line': parts[0],
            'mama': parts[1],
            'age': parts[2],
            'animal': parts[3],
            'genotype': parts[4]}
