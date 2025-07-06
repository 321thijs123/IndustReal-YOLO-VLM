# Preparing the dataset
1. Download the following zip files from [the IndustReal dataset page](https://data.4tu.nl/datasets/b008dd74-020d-4ea4-a8ba-7bb60769d224/2):
    * ``test_p1.zip``
    * ``test_p2.zip``
    * ``test_p3.zip``
    * ``train_p1.zip``
    * ``train_p2.zip``
    * ``train_p3.zip``
    * ``train_p4.zip``
    * ``val_p1.zip``
    * ``val_p2.zip``
    * ``assembly_state_detection_synthetic_data.zip.001``
    * ``assembly_state_detection_synthetic_data.zip.002``
2. Create a folder called ``real_data``, with sub folders ``train``, ``test``, and ``val``.
3. Extract the contents of each ``test_p*.zip``, ``train_p*.zip``, and ``val_p*.zip`` into the corresponding ``train``, ``test`` and, ``val`` folders.
4. Extract the contents of the ``assembly_state_detection_synthetic_data`` zips into a folder called ``synthetic_data``.
5. Seperate synthetic dataset files with labels:
    ```BASH
    python3 preprocess_synthetic.py
    ```
5. Convert the real dataset to YOLO format:
    ```BASH
    python3 create_dataset_real.py --dataset_input_folder "real_data"
    ```
6. Convert the synthetic dataset to YOLO format:
    ```BASH
    python3 create_dataset_synthetic.py --input_image_folder "labeled_synthetic_data/images/" --input_labels "labeled_synthetic_data/labels/"
    ```
7. Combine the real and synthetic dataset:
    ```BASH
    python3 create_dataset_hybrid.py --synth_dataset "synthetic_01" --real_dataset "real_dataset"
    ```

# YOLO Model
## Training
Weights for the trained YOLO model are included in this repository in ``yolo_weights/best.pt``.
These weights can be replicated by running:
```BASH
python3 train.py --dataset_name "combined_rgb"
```
## Generating outputs
The outputs used for the training and evaluation of the classifier are not included in this repository because they exceed the git file size limit, these outputs can be generated using:
```BASH
python3 generate_yolo_outputs.py
```
This will also generate images with bounding boxes for visualization in ``outputs/combined_rgb/yolo_test/``

# LLaVA Model
Due to the large amount of time required to generate LLaVA outputs, we only generate the outputs for a subset of the dataset, for this two scripts are provided. The first script generates outputs for uniformly sampled inputs from the entire dataset, the second script generates additional outputs for the non-zero classes, to improve the dataset balance.

The two different prompts that were evaluated are defined in the scripts below, by default prompt2 is commented out. The resulting outputs are already included in this repository in the ``outputs/combined_rgb/`` folder (``llava_test_prompt1.json``, ``llava_test_prompt1_addition.json``, ``llava_test_prompt2.json`` and ``llava_test_prompt2_addition.json``).

## Generating uniform LLaVA outputs
```BASH
python3 generate_llava_outputs.py
```

## Generating additional LLaVA outputs
```BASH
python3 generate_llava_outputs_addition.py
```

# Classification
## YOLO Only


## YOLO + VLM

