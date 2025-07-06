from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
import torch
from tqdm import tqdm
import json

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    quantization_config=quantization_config,
    device_map='auto'
)

import os
import numpy as np
from PIL import Image
from natsort import natsorted

text_prompt = """
You are analyzing a video that shows the assembly of a toy car. Your task is to provide a structured textual analysis that includes:

1. Which components are already installed.
2. Do NOT count duplicate or unused parts in piles unless they are clearly part of an assembly.

Use the information below to match the parts/components accurately:

PARTS (individual items):
Red:
- 5x nut
- 1x front wing (2x3 T Shape with sloped front)
- 1x cap nut (Elongated rounded cap)
- 1x pulley
- 4x washer

Black:
- 4x wheel

Gray:
- 5x Pin
- 2x Axle

White:
- 1x 1x1x4 beam
- 4x 1x4 straight flat plate
- 2x 2x6 L-shape flat plate
- 1x 1x2x3 U-shape

COMPONENTS (individual parts or assembled from multiple parts):
- Base = 1x1x4 beam + pin
- Front chassis = 2x 1x4 straight flat plate
- Front chassis pin = pin + nut
- Rear chassis = 2x 2x6 L-shape flat plate
- Short rear chassis = 2x 1x4 straight flat plate
- Front rear chassis pin = pin + nut
- Rear rear chassis pin = pin + nut
- Front bracket: 1x2x3 U-Shape
- Front bracket screw: pin + cap nut
- Front wheel assembly: wing + axle + 2x washer + 2x wheel + nut
- Rear wheel assembly: pulley + axle + 2x washer + 2x wheel + nut

ASSEMBLY:
The final product is a toy car that includes the components listed above.

Please analyze the video and return:
- A list of **components or parts installed**
- Be concise, minimize unneeded text, but maximize information.
- Note that parts may be visible but not installed, do not count those.

The response must not exceed 350 tokens so be concise.
"""

# text_prompt = """
# You are presented with a 6 second clip from a longer video of a person assembling a toy car.
# Your task is to describe which parts of the car have been assembled already.
# It may be that the car is not visible in the clip.
# The clip also likely contain parts that are not yet assembled, those may be in piles to the side or back of the table.
# You may describe unused parts, but make the distinction that they are not part of the assembly yet.
# Exclude text that does not contribute towards elaborating the assembly state: 
# - Do not include affirmation of this prompt
# - Do not add unncecsary text like "In the 6 second clip"
# Just immediately describe the assembly state.
# """

def get_clip(
    num_frames,
    start_idx,
    end_idx
):
    available = image_files[start_idx:end_idx]

    idxs = np.linspace(0, len(available) - 1, num=num_frames, dtype=int)
    selected = [available[i] for i in idxs]

    imgs = [np.array(Image.open(fp).convert("RGB")) for fp in selected]
    arr = np.stack(imgs, axis=0)                # shape = (N, H, W, 3), dtype=uint8

    return arr

def get_output(clip):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "video"},
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Move only once and track device explicitly
    inputs = processor([prompt], videos=[clip], padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=384)

    input_length = inputs['input_ids'].shape[-1]
    response = output[:, input_length:]
    response = processor.batch_decode(response, skip_special_tokens=True)

    # Free memory
    del inputs
    del output
    torch.cuda.empty_cache()

    return response

folder = './datasets/combined_rgb/images/test/'

image_files = natsorted(
    os.path.join(folder, f)
    for f in os.listdir(folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
)

num_frames = 16
frames_before = 32
frames_after = 32

interval = 100
offset= 0

json_dict = {"info": {
        "num_frames": num_frames,
        "frames_before": frames_before,
        "frames_after": frames_after,
        "interval": interval,
        "offset": offset,
        "prompt": text_prompt
    }, 
    "outputs": {}
}

output_folder = "outputs/combined_rgb/"
json_name = "llava_test_prompt1"

for center_frame in tqdm(range(frames_before, len(image_files) - frames_after, interval)):
    first_frame = center_frame - frames_before
    last_frame = center_frame + frames_after

    start_name = os.path.basename(image_files[first_frame])
    end_name = os.path.basename(image_files[last_frame])

    if start_name[:11] != end_name[:11]:
        continue
    
    center_name = os.path.basename(image_files[center_frame])

    clip = get_clip(num_frames, first_frame, last_frame)

    json_dict["outputs"][center_name] = get_output(clip)

    with open(os.path.join(output_folder, json_name + ".json"), 'w') as f:
        json.dump(json_dict, f, indent=4)

    
