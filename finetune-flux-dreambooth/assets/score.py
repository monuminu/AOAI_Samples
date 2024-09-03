import torch
import io
import os
import logging
import json
import math
import numpy as np
from base64 import b64encode
import requests
from PIL import Image, ImageDraw
from safetensors.torch import load_file
from azureml.contrib.services.aml_response import AMLResponse

from transformers import pipeline
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import AutoPipelineForText2Image, FluxPipeline
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from diffusers import DPMSolverMultistepScheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global pipe, refiner
    weights_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "pytorch_lora_weights.safetensors"
    )
    print("weights_path:", weights_path)
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.load_lora_weights(weights_path, use_safetensors=True)
    pipe.to(device)
    # refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    #                         "stabilityai/stable-diffusion-xl-refiner-1.0", 
    #                         torch_dtype=torch.float16, 
    #                         use_safetensors=True, 
    #                         variant="fp16"
    #                     )
    # refiner.to(device)
    logging.info("Init complete")


def get_image_object(image_url):
    """
    This function takes an image URL and returns an Image object.
    """
    response = requests.get(image_url)
    init_image = Image.open(io.BytesIO(response.content).convert("RGB"))
    return init_image

def prepare_response(images):
    """
    This function takes a list of images and converts them to a dictionary of base64 encoded strings.
    """
    ENCODING = 'utf-8'
    dic_response = {}
    for i, image in enumerate(images):
        output = io.BytesIO()
        image.save(output, format="JPEG")
        base64_bytes = b64encode(output.getvalue())
        base64_string = base64_bytes.decode(ENCODING)
        dic_response[f'image_{i}'] = base64_string
    return dic_response

def design(prompt, image=None, num_images_per_prompt=4, negative_prompt=None, strength=0.65, guidance_scale=7.5, num_inference_steps=50, seed=None, design_type='TXT_TO_IMG', mask=None, other_args=None):
    """
    This function takes various parameters like prompt, image, seed, design_type, etc., and generates images based on the specified design type. It returns a list of generated images.
    """
    generator = None
    if seed:
        generator = torch.manual_seed(seed)
    else:
        generator = torch.manual_seed(0)

    print('other_args', other_args)
    image = pipe(prompt=prompt, 
                 height=512,
                 width=768,
                 guidance_scale=guidance_scale,
                 output_type="latent", 
                 generator=generator).images[0]
    #image = refiner(prompt=prompt, image=image[None, :], generator=generator).images[0]    
    return [image]


def run(raw_data):
    """
     This function takes raw data as input, processes it, and calls the design function to generate images.
     It then prepares the response and returns it.
    """
    logging.info("Request received")
    print(f'raw data: {raw_data}')
    data = json.loads(raw_data)["data"]
    print(f'data: {data}')

    prompt = data['prompt']
    negative_prompt = data['negative_prompt']
    seed = data['seed']
    num_images_per_prompt = data['num_images_per_prompt']
    guidance_scale = data['guidance_scale']
    num_inference_steps = data['num_inference_steps']
    design_type = data['design_type']

    image_url = None
    mask_url = None
    mask = None
    other_args = None
    image = None
    strength = data['strength']

    if 'mask_image' in data:
        mask_url = data['mask_image']
        mask = get_image_object(mask_url)

    if 'other_args' in data:
        other_args = data['other_args']


    if 'image_url' in data:
        image_url = data['image_url']
        image = get_image_object(image_url)

    if 'strength' in data:
        strength = data['strength']

    with torch.inference_mode():
        images = design(prompt=prompt, image=image, 
                        num_images_per_prompt=num_images_per_prompt, 
                        negative_prompt=negative_prompt, strength=strength, 
                        guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                        seed=seed, design_type=design_type, mask=mask, other_args=other_args)
    
    preped_response = prepare_response(images)
    resp = AMLResponse(message=preped_response, status_code=200, json_str=True)

    return resp