import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
import os
import argparse

def gen_images(model_path, prompt):
    pipe = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None, torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    g_cuda = None

    g_cuda = torch.Generator(device='cuda')
    seed = 52361
    g_cuda.manual_seed(seed)

    negative_prompt = "" #@param {type:"string"}
    num_samples = 4 #@param {type:"number"}
    guidance_scale = 7 #@param {type:"number"}
    num_inference_steps = 50 #@param {type:"number"}
    height = 512 #@param {type:"number"}
    width = 512 #@param {type:"number"}

    with autocast("cuda"), torch.inference_mode():
        images = pipe(
            prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_samples,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=g_cuda
        ).images


    os.makedirs("generated", exist_ok=True)

    for i, img in enumerate(images):
        img.save(f"generated/output_{i}.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path to finetuned Stable Diffusion')
    parser.add_argument("--prompt")
    args = parser.parse_args()
    gen_images(args.model_path, args.prompt)