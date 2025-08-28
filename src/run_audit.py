import torch
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
import json
import os
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
PROMPT_FILE = str(BASE_DIR / "prompt_framework.json")
OUTPUT_DIR = str(BASE_DIR / "audit_results")
IMAGES_PER_PROMPT = int(os.getenv("IMAGES_PER_PROMPT", "20"))  # Override via env var for quick tests
MODEL_ID = os.getenv("MODEL_ID", "runwayml/stable-diffusion-v1-5")
NUM_INFERENCE_STEPS = int(os.getenv("NUM_INFERENCE_STEPS", "50"))  # 50 is default quality; reduce for speed
GUIDANCE_SCALE = float(os.getenv("GUIDANCE_SCALE", "7.5"))  # 7.5 is common default
DISABLE_SAFETY_CHECKER = os.getenv("DISABLE_SAFETY_CHECKER", "0") == "1"

# --- Setup ---
# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check for GPU availability and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Load Model ---
print("Loading Stable Diffusion model... (This may take a few minutes on the first run)")
# Use float16 for memory efficiency on GPU, fallback to float32 on CPU
dtype = torch.float16 if device == "cuda" else torch.float32
variant = "fp16" if device == "cuda" else None
pipeline = AutoPipelineForText2Image.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    variant=variant
).to(device)

# Use a faster scheduler for speed/quality balance
try:
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
except Exception:
    pass

# Optionally disable safety checker to reduce overhead (use with care)
if DISABLE_SAFETY_CHECKER:
    try:
        pipeline.safety_checker = None
    except Exception:
        pass

# Cleaner progress bars
pipeline.set_progress_bar_config(leave=False)
print("Model loaded successfully.")

# --- Load Prompt Framework ---
with open(PROMPT_FILE, 'r') as f:
    prompt_data = json.load(f)

# --- Run Generation Loop ---
for category, prompt_list in prompt_data.items():
    category_dir = os.path.join(OUTPUT_DIR, category)
    os.makedirs(category_dir, exist_ok=True)
    
    for prompt in prompt_list:
        # Create a safe folder name from the prompt
        prompt_folder_name = "".join(c for c in prompt if c.isalnum() or c in (' ', '_')).rstrip().replace(" ", "_")
        prompt_dir = os.path.join(category_dir, prompt_folder_name)
        os.makedirs(prompt_dir, exist_ok=True)
        
        print(f"\nGenerating {IMAGES_PER_PROMPT} images for prompt: '{prompt}'...")
        
        for i in range(IMAGES_PER_PROMPT):
            try:
                # Generate the image using the diffusers pipeline
                result = pipeline(
                    prompt,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                )
                images = getattr(result, "images", None)
                if images:
                    image = images[0]
                    # Save the image
                    image_path = os.path.join(prompt_dir, f"{i+1}.png")
                    image.save(image_path)
                    print(f"  Saved image {i+1}/{IMAGES_PER_PROMPT}")
                else:
                    print(f"  No image returned for iteration {i+1}")
            except Exception as e:
                print(f"  Error generating image {i+1}: {e}")

print("\nAudit generation complete.")
print(f"All images are saved in the '{OUTPUT_DIR}' directory.")