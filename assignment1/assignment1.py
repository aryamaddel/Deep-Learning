"""
Assignment: Study and implementation of Python libraries for
Deep Learning and Generative AI applications.
"""

import os
import numpy as np
import torch

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)


def banner(title):
    print("\n" + "=" * 72)
    print("  " + title)
    print("=" * 72)


# ---------------------------------------------------------
# 1. Verify installation and display library versions
# ---------------------------------------------------------
banner("1. LIBRARY VERSIONS (Installation Verified)")

libs = [
    ("numpy", "numpy"), ("pandas", "pandas"), ("matplotlib", "matplotlib"),
    ("scikit-learn", "sklearn"), ("torch", "torch"), ("tensorflow", "tensorflow"),
    ("cv2 (OpenCV)", "cv2"), ("transformers", "transformers"),
    ("diffusers", "diffusers"),
]
for name, alias in libs:
    try:
        mod = __import__(alias)
        version = getattr(mod, "__version__", "unknown")
        print(f"{name:<18} {version}")
    except Exception as e:
        print(f"{name:<18} ERROR: {e}")

print(f"PyTorch CUDA available : {torch.cuda.is_available()}")

# ---------------------------------------------------------
# 2. Tensor operations using NumPy and PyTorch
# ---------------------------------------------------------
banner("2. TENSOR OPERATIONS (NumPy + PyTorch)")

a_np = np.array([[1, 2, 3], [4, 5, 6]])
b_np = np.array([[7, 8, 9], [10, 11, 12]])
print("\nNumPy array a:\n", a_np)
print("NumPy addition (a+b):\n", a_np + b_np)
print("NumPy dot product (a . a.T):\n", a_np @ a_np.T)

a_t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b_t = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
print("\nPyTorch tensor a:\n", a_t)
print("PyTorch addition (a+b):\n", a_t + b_t)
print("PyTorch matmul (a . a.T):\n", a_t @ a_t.T)
print("PyTorch reshape (a -> 3x2):\n", a_t.reshape(3, 2))
print("PyTorch device:", a_t.device)

# ---------------------------------------------------------
# 3. Load a pre-trained MobileNet (TensorFlow/Keras) & infer
# ---------------------------------------------------------
banner("3. PRE-TRAINED MobileNetV2 (Image Classification)")

import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import matplotlib.image as mpimg
import requests

model = MobileNetV2(weights="imagenet")
print("Loaded MobileNetV2 with ImageNet weights.")

img_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg"
img_path = os.path.join(OUT, "sample.jpg")
r = requests.get(img_url, timeout=60)
with open(img_path, "wb") as f:
    f.write(r.content)

img = keras.utils.load_img(img_path, target_size=(224, 224))
img_arr = keras.utils.img_to_array(img)
img_arr = np.expand_dims(img_arr, axis=0)
img_arr = preprocess_input(img_arr)

preds = model.predict(img_arr, verbose=0)
labels = decode_predictions(preds, top=3)[0]
print(f"\nInput image saved to : {img_path}")
print("Top-3 predictions:")
for i, (imagenet_id, name, score) in enumerate(labels, 1):
    print(f"  {i}. {name:<20} {score*100:.2f}%")

# ---------------------------------------------------------
# 4. Generate text using Hugging Face Transformers
# ---------------------------------------------------------
banner("4. TEXT GENERATION (Hugging Face Transformers)")

from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
prompt = "The future of artificial intelligence is"
output = generator(prompt, max_new_tokens=30, num_return_sequences=1)[0]
print(f"\nPrompt : {prompt}")
print("Generated :", output["generated_text"])

# ---------------------------------------------------------
# 5. Image generation using Diffusers (Stable Diffusion)
# ---------------------------------------------------------
banner("5. IMAGE GENERATION (Hugging Face Diffusers)")

from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "segmind/tiny-sd", torch_dtype=torch.float32, safety_checker=None
)
img = pipe("a cute robot painting a sunset beside a lake", num_inference_steps=15).images[0]
gen_path = os.path.join(OUT, "generated_image.png")
img.save(gen_path)
print(f"Generated image saved to : {gen_path}")

print("\n" + "=" * 72)
print("  ALL STEPS DONE.")
print("=" * 72)
