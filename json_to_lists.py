import json

id_to_label = [
    # One additional bit at the end to indicate error state 23
    [0,0,0,0,0,0,0,0,0,0,0,0], # 0
    [1,0,0,0,0,0,0,0,0,0,0,0], # 1
    [1,0,0,1,0,0,1,0,0,0,0,0], # 2
    [1,0,0,1,0,1,0,0,0,0,0,0], # 3
    [1,0,0,1,0,1,1,0,0,0,0,0], # 4
    [1,1,1,0,0,0,0,0,0,0,0,0], # 5
    [1,1,1,1,0,0,1,0,0,0,0,0], # 6
    [1,1,1,1,0,1,0,0,0,0,0,0], # 7
    [1,1,1,1,0,1,1,0,0,0,0,0], # 8
    [1,1,1,1,0,1,1,1,1,0,0,0], # 9
    [1,1,1,1,0,1,1,1,1,1,0,0], # 10
    [1,1,1,1,0,1,1,0,0,0,1,0], # 11
    [1,1,1,1,0,1,1,1,1,0,1,0], # 12
    [1,1,1,1,0,1,1,1,1,1,1,0], # 13
    [1,1,1,1,0,1,0,1,1,1,1,0], # 14
    [1,1,1,1,0,0,1,1,1,1,1,0], # 15
    [1,1,1,1,0,0,1,1,1,1,0,0], # 16
    [1,1,1,1,0,1,0,1,1,1,0,0], # 17
    [1,1,1,0,0,0,0,1,1,1,0,0], # 18
    [1,1,1,0,1,1,0,1,1,1,0,0], # 19
    [1,1,1,0,1,0,1,1,1,1,0,0], # 20
    [1,1,1,0,1,1,1,1,1,1,0,0], # 21
    [1,1,1,0,1,1,1,1,1,1,1,0], # 22
    [0,0,0,0,0,0,0,0,0,0,0,1], # 23
]

def read_yolo_json(json_path):
    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)

    names = []
    datas = []

    for name, classes in json_data.items():
        data = []

        for cls in classes:
            data.extend(cls["box"])
            data.append(cls["conf"])
        
        names.append(name)
        datas.append(data)
    
    return names, datas

def get_labels(file_names):
    return [id_to_label[int(file_name[-6:-4])] for file_name in file_names]

names, yolo_data = read_yolo_json("outputs/combined_rgb/test.json")
labels = get_labels(names)