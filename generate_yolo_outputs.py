from ultralytics import YOLO
import os
from tqdm import tqdm
import json
from copy import deepcopy

input_folder = "datasets/combined_rgb/images/"
output_folder = "outputs/combined_rgb/"
phases = ["test"]
weights_file = "yolo_weights/best.pt"

model = YOLO(weights_file)
batch_size = 256

num_classes = len(model.names)

new_box_entry = [{"box": [0.0, 0.0, 0.0, 0.0], "conf": 0.0} for _ in range(num_classes)]

def batches(file_list):
    batches = []
    
    i = 0

    while i < len(file_list) - batch_size:
        batches.append(file_list[i:i+batch_size])
        i += batch_size

    batches.append(file_list[i:])

    return batches

for phase in phases:
    boxes_dict = {}
    
    print(f"Phase: {phase}")
    phase_input = os.path.join(input_folder, phase)
    phase_output = os.path.join(output_folder, "yolo_" + phase)
    os.makedirs(phase_output, exist_ok=True)

    phase_files = [os.path.join(phase_input, f) for f in os.listdir(phase_input)]

    for file_batch in tqdm(batches(phase_files)):
        results = model(file_batch, verbose=False, conf=0.0001)

        for result, file in zip(results, file_batch):
            file_name =  os.path.basename(file)

            boxes_dict[file_name] = deepcopy(new_box_entry)

            for box in result.boxes:
                cls = round(box.cls.item())
                conf = box.conf.item()

                if (conf > boxes_dict[file_name][cls]["conf"]):
                    boxes_dict[file_name][cls]["box"] = box.xyxyn[0].tolist()
                    boxes_dict[file_name][cls]["conf"] = conf

            result.save(os.path.join(phase_output, file_name))

    with open(os.path.join(output_folder, "yolo_" + phase + ".json"), 'w') as f:
        json.dump(boxes_dict, f, indent=4)
    