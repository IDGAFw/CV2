import json, time
import numpy as np, torch, gradio as gr
from torchvision import transforms
from a2_common import get_device, sync, ViTClassifier, predict_with_rollout, overlay, VIT_MEAN, VIT_STD

classes = json.load(open('classes.json'))
device = get_device()
model = ViTClassifier(len(classes))
model.load_state_dict(torch.load('vit_scene.pth', map_location='cpu'))
model.to(device).eval()
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(VIT_MEAN, VIT_STD)])

predict_with_rollout(model, torch.zeros(1, 3, 224, 224, device=device)); sync(device)  # warm-up
lat = []


def predict(img):
    if img is None: return None, None, ""
    img = img.convert('RGB')
    x = tf(img).unsqueeze(0).to(device)
    sync(device); t = time.perf_counter()
    logits, mask = predict_with_rollout(model, x)
    sync(device); ms = (time.perf_counter() - t) * 1000
    lat.append(ms)
    probs = logits.softmax(-1)[0].cpu().numpy()
    info = (f"**Latency:** {ms:.0f} ms (avg {np.mean(lat):.0f} ms over {len(lat)} runs)  \n"
            f"**Device:** {device.type} · PyTorch {torch.__version__} · ViT-B/16 (fine-tuned)")
    return {c: float(p) for c, p in zip(classes, probs)}, overlay(img, mask[0]), info


with gr.Blocks(title="Scene Classifier — ViT + Attention Rollout") as demo:
    gr.Markdown("# Scene Classifier (ViT-B/16 + Attention Rollout)\nUpload a photo → prediction → heatmap of where the model looked.")
    with gr.Row():
        inp = gr.Image(type="pil", label="Upload / camera", sources=["upload", "webcam"])
        with gr.Column():
            lab = gr.Label(num_top_classes=len(classes), label="Prediction")
            heat = gr.Image(label="Attention rollout")
    info = gr.Markdown()
    gr.Button("Predict", variant="primary").click(predict, inp, [lab, heat, info])
    inp.change(predict, inp, [lab, heat, info])
    gr.Markdown("**Ethics & disclosure:** images are processed locally and not stored. The heatmap shows attention flow, "
                "not a causal explanation. The dataset is self-collected; avoid uploading photos of people without their consent. "
                "This demo contains no synthetic/generated media.")

demo.launch()
