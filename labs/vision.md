# Computer Vision: Inside the Black Box Lab
**Inspired by Russell & Norvig, *Artificial Intelligence: A Modern Approach*, Ch. 25 — Computer Vision**

---

## Learning Objectives

By the end of this lab, you will be able to:

1. **Identify** what visual features are learned at different depths of a convolutional neural network
2. **Distinguish** between early-layer and late-layer feature representations in a CNN
3. **Explain** how gradient-based attribution (Grad-CAM) reveals which image regions drive model decisions
4. **Analyze** the relationship between a model's attention and its classification reasoning
5. **Evaluate** the implications for real-world AI deployment

---

## Lab Overview

This lab takes you inside a trained convolutional neural network — not to build one, but to *see* what it has learned. You will run three progressively revealing experiments:

- **Exercise 1–3**: Visualizing CNN feature maps across layers to observe hierarchical feature learning
- **Exercise 4–5**: Using Grad-CAM heatmaps to understand where the model "looks" when making predictions
- **Exercise 6–7**: Crafting adversarial examples to probe the fragility of learned representations

**Pedagogical Approach:** You will run fully implemented code and carefully observe outputs. Your goal is to build intuition about how deep learning systems perceive and process images — and where they can fail.

### Setup

```bash
uv add torch torchvision matplotlib opencv-python grad-cam torchattacks
```

**Python:** 3.10+  
**Domain:** ImageNet-pretrained ResNet18 applied to natural images. ResNet18 is a classic, well-understood architecture that maps cleanly to textbook CNN concepts.

---

#### uv

I highly recommend uv (https://docs.astral.sh/uv/). It (according to their docs):

- 🚀 Is a single tool to replace pip, pip-tools, pipx, poetry, pyenv, twine, virtualenv, and more.
- ⚡️ Is 10-100x faster than pip.
- 🗂️ Provides comprehensive project management, with a universal lockfile.
- ❇️ Runs scripts, with support for inline dependency metadata.
- 🐍 Installs and manages Python versions.

---

#### Jupyter Notebook

This lab is designed to be run in a Jupyter notebook environment, because the examples build progressively.

Select the virtual environment created by `uv`` (`cs430`) as the kernel for your Jupyter notebook.

Paste the code for each exercise in a new code cell.

If you can't use Jupyter Notebook for whatever reason, just build up a regular Python program, and ignore output from earlier exercises.

#### Submission 

Make sure to record your answers to *all* reflections to submit at the end of the lab!

---

## Exercise 1: Loading a Pretrained CNN and Inspecting Its Architecture

### Description

Before visualizing anything, we need to understand the *structure* of the model we're working with. This exercise loads a pretrained ResNet18 and prints its layers so you can see the hierarchy of convolutions, activations, and pooling operations.

### Key Concepts

- **Convolutional layer**: Applies learned filters to detect local patterns (edges, textures, shapes)
- **Feature map**: The output of a convolutional layer — a 2D grid showing where a particular pattern was detected
- **Pretrained model**: A network already trained on ImageNet (1.2M images, 1000 classes); its weights encode learned visual knowledge
- **Hierarchical representation**: The idea that early layers detect simple features and later layers combine them into complex concepts
- **Layer depth**: Refers to how many transformations an input has passed through; deeper = more abstract

### Task

Run the code and observe the printed architecture. Pay attention to how the layers are organized into sequential blocks (`layer1` through `layer4`). Notice how the number of channels *increases* as you go deeper (64 → 128 → 256 → 512).

```python
import torch
import torchvision.models as models

# Load pretrained ResNet18 — weights learned from 1.2M ImageNet images
model = models.resnet18(pretrained=True)
model.eval()  # Set to evaluation mode (disables dropout, etc.)

print("=== ResNet18 Architecture ===\n")
for name, module in model.named_children():
    print(f"[{name}]  →  {module.__class__.__name__}")

print("\n=== Layer 1 Detail ===")
print(model.layer1)

print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**Watch for:**
- The names `layer1`, `layer2`, `layer3`, `layer4` — these are the four residual "stages"
- The `conv1` at the top — this is where raw pixels first enter the network
- How each stage contains multiple `BasicBlock` modules

### No Reflection Questions!

---

## Exercise 2: Extracting and Visualizing Layer 1 Feature Maps

### Description

Now we extract the actual feature maps from the *first* convolutional layer and display them as images. Layer 1 operates directly on raw pixels, so its filters should detect the most primitive visual signals: oriented edges, color gradients, and blobs.

### Key Concepts

- **Hook**: A PyTorch mechanism for intercepting intermediate layer outputs without modifying the model
- **Filter/kernel**: A small matrix of weights that slides over the image to produce one feature map
- **Edge detection**: Identifying boundaries between regions of different intensity or color — the foundational visual primitive
- **Activation**: The output value at a location in a feature map; high activation = strong match to the filter's pattern
- **Channel**: Each filter in a convolutional layer produces one output channel (feature map)

### Task

Run the code with any `.jpg` image in the same directory (rename it `input.jpg`), or use the URL loading variant. Observe the 16 feature maps printed from layer 1. Look for edges, outlines, and directional gradients in the outputs.

```python
import torch
import torchvision.models as models
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

# --- Load model and register a hook on the first conv layer ---
model = models.resnet18(pretrained=True)
model.eval()

feature_maps = {}

def save_output(name):
    def hook(module, input, output):
        feature_maps[name] = output.detach()
    return hook

model.layer1.register_forward_hook(save_output("layer1"))

# --- Load and preprocess an image ---
# Wikimedia requires a User-Agent header or it returns a non-image error page
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
headers = {"User-Agent": "Mozilla/5.0 (cv-lab-student-project)"}
img = Image.open(BytesIO(requests.get(url, headers=headers).content)).convert("RGB")

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
tensor = transform(img).unsqueeze(0)  # Add batch dimension

# --- Forward pass ---
with torch.no_grad():
    _ = model(tensor)

# --- Visualize first 16 feature maps from layer1 ---
fmaps = feature_maps["layer1"][0]  # Shape: [64, 56, 56]
print(f"Layer 1 output shape: {fmaps.shape}  (channels × height × width)")

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
fig.suptitle("Layer 1 Feature Maps — What early filters detect", fontsize=14)

for i, ax in enumerate(axes.flat):
    ax.imshow(fmaps[i].numpy(), cmap="viridis")
    ax.set_title(f"Filter {i}", fontsize=8)
    ax.axis("off")

plt.tight_layout()
plt.savefig("layer1_features.png", dpi=100)
plt.show()
print("Saved: layer1_features.png")
```

**Watch for:** Some feature maps will show clear edge responses (light lines on dark backgrounds). Others will look like blobs detecting color or texture. Notice that most filters respond to *local* structure — they don't see the whole image at once.

### Reflection Questions

**Q1.** Describe two or three distinct visual patterns you observe across the 16 feature maps. What kind of image structures (edges, textures, colors) does Layer 1 appear to be responding to?

**Q2.** Some feature maps appear almost entirely dark or uniform. What does a near-zero activation mean in terms of what the filter "found" in this particular image? What would cause a filter to activate strongly?

---

## Exercise 3: Comparing Layer 1 vs. Layer 4 Feature Maps

### Description

This is the core observation exercise. We extract feature maps from both the *shallowest* and *deepest* residual stages and compare them side by side. This directly demonstrates hierarchical feature learning — one of the foundational ideas in deep learning and perception.

### Key Concepts

- **Hierarchical feature learning**: The principle that networks learn increasingly abstract representations as depth increases
- **Receptive field**: The region of the original input image that influences a particular neuron; grows larger in deeper layers
- **Semantic abstraction**: The shift from pixel-level features (edges) to concept-level features (dog faces, wheel shapes)
- **Feature entanglement**: In deep layers, individual channels no longer correspond to interpretable single features
- **Spatial resolution trade-off**: Deeper layers have smaller spatial maps but richer per-location representations

### Task

Run the code and examine the two sets of feature maps. Layer 1 maps should look structured and interpretable. Layer 4 maps will look chaotic — but they encode far more semantic information.

```python
import torch
import torchvision.models as models
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

model = models.resnet18(pretrained=True)
model.eval()

maps = {}

for layer_name in ["layer1", "layer4"]:
    getattr(model, layer_name).register_forward_hook(
        lambda m, i, o, n=layer_name: maps.update({n: o.detach()})
    )

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
img = Image.open(BytesIO(requests.get(url, headers={"User-Agent": "Mozilla/5.0 (cv-lab-student-project)"}).content)).convert("RGB")
tensor = T.Compose([
    T.Resize((224, 224)), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])(img).unsqueeze(0)

with torch.no_grad():
    _ = model(tensor)

fig, axes = plt.subplots(2, 8, figsize=(16, 5))
fig.suptitle("Layer 1 (top) vs Layer 4 (bottom) — Shallow vs Deep Features", fontsize=13)

for i in range(8):
    axes[0, i].imshow(maps["layer1"][0][i].numpy(), cmap="plasma")
    axes[0, i].set_title(f"L1-F{i}", fontsize=7)
    axes[0, i].axis("off")

    axes[1, i].imshow(maps["layer4"][0][i].numpy(), cmap="plasma")
    axes[1, i].set_title(f"L4-F{i}", fontsize=7)
    axes[1, i].axis("off")

print(f"Layer 1 feature map shape: {maps['layer1'].shape}")
print(f"Layer 4 feature map shape: {maps['layer4'].shape}")

plt.tight_layout()
plt.savefig("layer_comparison.png", dpi=100)
plt.show()
```

**Watch for:** Layer 4 maps are `7×7` (vs. Layer 1's `56×56`). Each "pixel" in a Layer 4 map corresponds to a `32×32` patch of the original image. The patterns will be coarser and less obviously related to specific edges.

### Reflection Questions

**Q3.** How do the Layer 4 feature maps differ visually from Layer 1 maps? What does this tell you about the nature of the representations learned at different depths?

**Q4.** If you used a completely different image (e.g., a car instead of a dog), which layer's feature maps would change more dramatically — Layer 1 or Layer 4? Justify your answer based on what each layer has learned.

---

## Exercise 4: Grad-CAM — Where Does the Model Look?

### Description

Grad-CAM (Gradient-weighted Class Activation Mapping) uses gradients flowing back from the predicted class to identify which spatial regions of the image most influenced that prediction. It produces a heatmap overlaid on the original image, answering the question: *"What did the model pay attention to?"*

### Key Concepts

- **Gradient**: The direction and magnitude of change in the loss with respect to intermediate activations
- **Class activation map**: A spatial map indicating which image regions contribute most to a specific class prediction
- **Grad-CAM**: Computes the weighted average of feature maps using gradients as importance weights
- **Attribution**: Assigning credit (or blame) to parts of the input for the model's output
- **Saliency**: Which parts of an input are most "salient" (important) to the model's decision

### Task

Run the code and examine the heatmap overlay. The red/warm regions are where the model concentrated its attention. Check whether the highlighted region corresponds to the actual object or to background context.

```python
import torch
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import requests
from io import BytesIO

# Load model
model = models.resnet18(pretrained=True)
model.eval()

# Target the last convolutional layer — richest spatial features
target_layers = [model.layer4[-1]]

# Load image
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
img_pil = Image.open(BytesIO(requests.get(url, headers={"User-Agent": "Mozilla/5.0 (cv-lab-student-project)"}).content)).convert("RGB").resize((224, 224))
img_np = np.array(img_pil) / 255.0  # Normalized for overlay

tensor = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])(img_pil).unsqueeze(0)

# Run Grad-CAM
cam = GradCAM(model=model, target_layers=target_layers)
grayscale_cam = cam(input_tensor=tensor, targets=None)  # targets=None → top predicted class

# Get predicted class name
with torch.no_grad():
    output = model(tensor)
    pred_idx = output.argmax(1).item()

# Load ImageNet class labels
labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = requests.get(labels_url, headers={"User-Agent": "Mozilla/5.0 (cv-lab-student-project)"}).text.strip().split("\n")
print(f"Predicted class: {labels[pred_idx]}  (index {pred_idx})")
print(f"Confidence: {torch.softmax(output, dim=1)[0][pred_idx].item():.2%}")

# Overlay heatmap
visualization = show_cam_on_image(img_np.astype(np.float32), grayscale_cam[0], use_rgb=True)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
axes[0].imshow(img_pil); axes[0].set_title("Original Image"); axes[0].axis("off")
axes[1].imshow(grayscale_cam[0], cmap="jet"); axes[1].set_title("Grad-CAM Heatmap"); axes[1].axis("off")
axes[2].imshow(visualization); axes[2].set_title(f"Overlay — Predicted: {labels[pred_idx]}"); axes[2].axis("off")

plt.tight_layout()
plt.savefig("gradcam.png", dpi=100)
plt.show()
```

**Watch for:** Is the red region covering the dog's face/body, or is it landing on the background grass/sky? Perfect alignment = the model learned the right features. Misalignment = the model may be using spurious correlations.

### Reflection Questions

**Q5.** Describe the spatial pattern of the Grad-CAM heatmap. Does the model appear to focus on the object of interest or on surrounding context? What are the implications for model trustworthiness?

**Q6.** Grad-CAM uses gradients from a *specific class* to generate the heatmap. What would happen to the heatmap if you asked it to explain the prediction for a *wrong* class (e.g., generating a heatmap for "cat" when the image contains a dog)?

---

## Exercise 5: Grad-CAM Across Multiple Images — Shortcut Learning

### Description

A single Grad-CAM can be coincidentally correct. This exercise runs Grad-CAM on multiple images to look for patterns in what the model attends to — and to surface potential *shortcut learning*, where a model exploits dataset biases rather than learning the true concept.

### Key Concepts

- **Shortcut learning**: A model learns a spurious correlation in training data (e.g., "polar bears appear on snow") rather than the actual concept
- **Distribution shift**: When test data differs from training data, shortcut-reliant models fail unexpectedly
- **In-context vs. out-of-context objects**: Testing a model on objects in unusual settings reveals whether it learned the object or its typical background
- **Model debugging**: Using interpretability tools to identify failure modes before deployment
- **Spurious correlation**: A statistical relationship in training data that does not reflect a causal relationship

### Task

Run the code for at least two different images. Compare where the model looks for each. Note any cases where the model confidently predicts the right label but attends to the wrong region.

```python
import torch
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import requests
from io import BytesIO

model = models.resnet18(pretrained=True)
model.eval()
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

headers = {"User-Agent": "Mozilla/5.0 (cv-lab-student-project)"}
labels = requests.get(
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", headers=headers
).text.strip().split("\n")

# Try different images — swap URLs to explore
test_images = {
    "Dog on grass": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg",
    "Pier":  "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Image_created_with_a_mobile_phone.png/320px-Image_created_with_a_mobile_phone.png",
}

transform = T.Compose([
    T.Resize((224, 224)), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

fig, axes = plt.subplots(len(test_images), 3, figsize=(14, 5 * len(test_images)))

for row, (label, url) in enumerate(test_images.items()):
    img_pil = Image.open(BytesIO(requests.get(url, headers={"User-Agent": "Mozilla/5.0 (cv-lab-student-project)"}).content)).convert("RGB").resize((224, 224))
    img_np = np.array(img_pil) / 255.0
    tensor = transform(img_pil).unsqueeze(0)

    grayscale_cam = cam(input_tensor=tensor, targets=None)

    with torch.no_grad():
        output = model(tensor)
    pred_idx = output.argmax(1).item()
    confidence = torch.softmax(output, dim=1)[0][pred_idx].item()

    overlay = show_cam_on_image(img_np.astype(np.float32), grayscale_cam[0], use_rgb=True)

    axes[row, 0].imshow(img_pil); axes[row, 0].set_title(label); axes[row, 0].axis("off")
    axes[row, 1].imshow(grayscale_cam[0], cmap="jet"); axes[row, 1].set_title("Heatmap"); axes[row, 1].axis("off")
    axes[row, 2].imshow(overlay)
    axes[row, 2].set_title(f"Pred: {labels[pred_idx]} ({confidence:.1%})")
    axes[row, 2].axis("off")

    print(f"{label} → Predicted: {labels[pred_idx]}  ({confidence:.1%})")

plt.tight_layout()
plt.savefig("gradcam_multi.png", dpi=100)
plt.show()
```

**Watch for:** Cases where the model attends to *background* rather than the foreground object. Note the confidence score — high confidence does not imply correct reasoning.

### Reflection Questions

**Q7.** If a model correctly classifies a "polar bear" image but Grad-CAM shows it attended primarily to the snow background, what does this tell us about what the model actually learned? How might this model perform on a polar bear image in a zoo?

---

## Exercise 6: Adversarial Examples — Breaking the Model with Noise

### Description

Adversarial examples are inputs crafted with small, carefully computed perturbations that cause a model to misclassify an image — even when the perturbation is completely invisible to the human eye. This exercise demonstrates the FGSM (Fast Gradient Sign Method) attack, one of the earliest and most instructive adversarial attacks.

### Key Concepts

- **Adversarial example**: An input modified by a small perturbation designed to fool a classifier while remaining imperceptible to humans
- **FGSM (Fast Gradient Sign Method)**: Computes the gradient of the loss with respect to the *input pixels* and nudges each pixel in the direction that increases the loss
- **Epsilon (ε)**: The maximum allowed perturbation magnitude; controls the attack strength
- **Perturbation norm**: A measure (e.g., L∞ or L2) of how much the input was changed
- **Threat model**: The assumptions about what an adversary can and cannot do (in FGSM, the adversary can modify all pixels within ε)

### Task

Run the code and observe the original, perturbation, and adversarial images side by side. Note that the perturbation looks like visual noise to a human but completely changes the model's prediction. Increase `epsilon` and observe how both the visual distortion and misclassification confidence change.

```python
import torch
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

model = models.resnet18(pretrained=True)
model.eval()

labels = requests.get(
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
).text.strip().split("\n")

# Load image
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
img_pil = Image.open(BytesIO(requests.get(url, headers={"User-Agent": "Mozilla/5.0 (cv-lab-student-project)"}).content)).convert("RGB").resize((224, 224))

transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

tensor = transform(img_pil).unsqueeze(0).requires_grad_(True)  # Enable gradients on INPUT

# --- Forward pass: get original prediction ---
output = model(tensor)
orig_class = output.argmax(1).item()
print(f"Original prediction: {labels[orig_class]}  ({torch.softmax(output,1)[0][orig_class].item():.1%})")

# --- FGSM Attack ---
epsilon = 0.02  # Try 0.01, 0.05, 0.1 and observe changes
loss = torch.nn.CrossEntropyLoss()(output, torch.tensor([orig_class]))
loss.backward()  # Compute gradients with respect to INPUT pixels

# Perturb: move each pixel in the direction that increases loss
perturbation = epsilon * tensor.grad.sign()
adversarial = tensor + perturbation

# --- Classify adversarial image ---
with torch.no_grad():
    adv_output = model(adversarial)
adv_class = adv_output.argmax(1).item()
adv_conf = torch.softmax(adv_output, 1)[0][adv_class].item()
print(f"Adversarial prediction: {labels[adv_class]}  ({adv_conf:.1%})")
print(f"L-inf perturbation norm: {perturbation.abs().max().item():.4f}")

# --- Visualize ---
def to_displayable(t):
    """Denormalize tensor back to [0,1] for display"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return (t.squeeze() * std + mean).clamp(0, 1).permute(1,2,0).detach().numpy()

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
axes[0].imshow(to_displayable(tensor)); axes[0].set_title(f"Original\n{labels[orig_class]}"); axes[0].axis("off")

# Amplify perturbation 10× for visibility
perturb_vis = (perturbation.squeeze().permute(1,2,0).detach().numpy() * 10 + 0.5).clip(0,1)
axes[1].imshow(perturb_vis); axes[1].set_title(f"Perturbation (×10 amplified)\nε = {epsilon}"); axes[1].axis("off")

axes[2].imshow(to_displayable(adversarial)); axes[2].set_title(f"Adversarial\n{labels[adv_class]} ({adv_conf:.1%})"); axes[2].axis("off")

plt.tight_layout()
plt.savefig("adversarial.png", dpi=100)
plt.show()
```

**Watch for:** The amplified perturbation looks like patterned noise. The adversarial image looks nearly identical to the original. Yet the model produces a completely different — and often confidently wrong — prediction.

### Reflection Questions

**Q8.** Look at the amplified perturbation image. Does it look like anything meaningful to you? Now consider that the model finds this pattern highly informative — what does this reveal about the difference between how humans and CNNs represent images?

---

## Exercise 7: Perturbation Norms and the Geometry of Adversarial Space

### Description

This final exercise sweeps across multiple epsilon values and plots how classification confidence and perturbation norm change together. This quantitative view reveals the *geometry* of adversarial vulnerability — how far from the original image the model's decision boundary lies.

### Key Concepts

- **Decision boundary**: The hypersurface in pixel space that separates one class's region from another's
- **L∞ norm**: The maximum absolute change across all pixel values; measures "worst-case" per-pixel distortion
- **L2 norm**: Euclidean distance between original and adversarial images; measures total image change
- **Robustness**: A model's ability to maintain correct predictions under input perturbation
- **Accuracy-robustness trade-off**: Models optimized purely for accuracy on clean data often sacrifice robustness

### Task

Run the code and observe the two plots. Note at what epsilon value the model's confidence in the *correct* class drops sharply, and what the L2 norm is at that point. This tells you how close the original image is to the model's decision boundary.

```python
import torch
import torchvision.models as models
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

model = models.resnet18(pretrained=True)
model.eval()

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
img_pil = Image.open(BytesIO(requests.get(url, headers={"User-Agent": "Mozilla/5.0 (cv-lab-student-project)"}).content)).convert("RGB").resize((224, 224))
transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
tensor = transform(img_pil).unsqueeze(0)

# Get original class
with torch.no_grad():
    orig_output = model(tensor)
orig_class = orig_output.argmax(1).item()

epsilons = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2]
correct_confs, adv_confs, l2_norms, linf_norms = [], [], [], []

for eps in epsilons:
    t = tensor.clone().requires_grad_(True)
    out = model(t)
    loss = torch.nn.CrossEntropyLoss()(out, torch.tensor([orig_class]))
    loss.backward()

    with torch.no_grad():
        perturb = eps * t.grad.sign()
        adv = t + perturb
        adv_out = model(adv)
        probs = torch.softmax(adv_out, 1)[0]

        correct_confs.append(probs[orig_class].item())
        adv_confs.append(probs.max().item())
        l2_norms.append(perturb.norm(p=2).item())
        linf_norms.append(perturb.abs().max().item())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(epsilons, correct_confs, "b-o", label="Confidence in correct class")
axes[0].plot(epsilons, adv_confs,    "r-o", label="Confidence in predicted (adv) class")
axes[0].set_xlabel("Epsilon (attack strength)")
axes[0].set_ylabel("Softmax Confidence")
axes[0].set_title("Classification Confidence vs. Attack Strength")
axes[0].legend(); axes[0].grid(True)

axes[1].plot(epsilons, l2_norms,   "g-o", label="L2 norm")
axes[1].plot(epsilons, linf_norms, "m-o", label="L∞ norm")
axes[1].set_xlabel("Epsilon")
axes[1].set_ylabel("Perturbation Magnitude")
axes[1].set_title("Perturbation Norms vs. Attack Strength")
axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
plt.savefig("robustness_curves.png", dpi=100)
plt.show()

print(f"Original class confidence at ε=0: {correct_confs[0]:.1%}")
print(f"Confidence drops below 50% at ε ≈ {epsilons[[c < 0.5 for c in correct_confs].index(True)]}")
```

**Watch for:** The epsilon value at which the correct-class confidence "collapses." Note that the L2 norm grows slowly — meaning the image hasn't changed much in absolute terms when the model already fails.

### Reflection Questions

**Q9.** At what epsilon value does the model lose confidence in the correct class below 50% in your run? What does this tell you about how close the original image is to the decision boundary in pixel space?

**Q10.** Consider a safety-critical application like medical imaging diagnosis or autonomous driving. Based on your observations across Exercises 6 and 7, what specific risks do adversarial examples pose, and what would you want to know about a deployed model's adversarial robustness before trusting it?

---

## Summary and Key Takeaways

This lab revealed three interconnected truths about modern deep learning systems:

**Hierarchical feature learning** is real and observable. Layer 1 filters detect edges and gradients; Layer 4 encodes rich, abstract representations tied to object semantics. This mirrors the classical AI idea that perception involves successive transformations from raw signals to meaningful symbols — but in a CNN, these representations are *learned* rather than hand-engineered.

**Model predictions and model reasoning can diverge.** Grad-CAM showed that a model can classify an image correctly for the wrong reasons — attending to background context rather than the object itself. This is the practical manifestation of shortcut learning, and it has profound implications for deployment in novel environments. Accuracy metrics alone are insufficient for evaluating AI systems.

**Deep CNNs operate in a fundamentally different perceptual space than humans.** Adversarial examples — invisible to human eyes but catastrophically confusing to models — reveal that CNNs do not perceive images the way we do. Their decision boundaries are jagged and close to natural images in directions humans would never explore. This fragility is not a bug that will be patched; it reflects a deep structural difference between human and machine perception.

Together, these observations connect to a central theme in Russell & Norvig: building AI systems that are not just accurate, but *interpretable*, *robust*, and *trustworthy* — systems whose behavior we can understand, predict, and correct.

---

## Submission Instructions

Create a new **public** Github Repository called `cs430`, upload your local `cs430` folder there including all code from this lab and:

Create `lab_vision_results.md`:

```markdown
# Names: Your names here
# Lab: lab7 (Vision)
# Date: Today's date
```

And your answers to all reflection questions above. Each answer should be 2-5 sentences that demonstrate your understanding of the concepts through the lens of the exercises you ran.

Email the GitHub repository web link to me at `chike.abuah@wallawalla.edu`

*If you're concerned about privacy* 

You can make a **private** Github Repo and add me as a collaborator, my username is `abuach`.

Congrats, you're done with the seventh lab!
