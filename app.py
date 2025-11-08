import gradio as gr
import torch
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
import os
import time

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

# --- 1. Global Configuration & Model Loading ---

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
MODEL_CONFIG_PATH = 'configs/refldm.yaml'
MODEL_CKPT_PATH = 'ckpts/refldm.ckpt'
VAE_CKPT_PATH = 'ckpts/vqgan.ckpt'

def load_model(config_path, ckpt_path, vae_path):
    """Loads the model onto the device only once."""
    print("Loading model...")
    config = OmegaConf.load(config_path)
    config.model.params.first_stage_config.params.ckpt_path = vae_path
    
    for k in ['ckpt_path', 'perceptual_loss_config']:
        config.model.params.pop(k, None)
        
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
    
    model = model.to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}")
    return model

# Load the model when the script starts
model = load_model(MODEL_CONFIG_PATH, MODEL_CKPT_PATH, VAE_CKPT_PATH)
sampler = DDIMSampler(model, print_tqdm=False, schedule='uniform_trailing')

# --- 2. Core Image Processing Functions ---

def read_and_normalize_image(image, size=(512, 512)):
    """Normalizes a PIL image for the model."""
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image")

    image = image.convert("RGB")
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    
    tensor = pil_to_tensor(image)
    tensor = tensor / 127.5 - 1.0  # [0, 255] to [-1, 1]
    return tensor

def process_and_restore(lq_image, ref_images, progress=gr.Progress(track_tqdm=True)):
    """The main function to run inference when the button is clicked."""
    # Input validation
    if lq_image is None:
        raise gr.Error("请上传一张低质量图片 (LQ Image)！")
    if not ref_images or len(ref_images) > 4:
        raise gr.Error("请上传 1 到 4 张参考图片 (Reference Images)！")

    progress(0, desc="准备输入数据...")
    time.sleep(0.2)

    try:
        # Prepare condition
        lq_tensor = read_and_normalize_image(lq_image)
        ref_tensors = [read_and_normalize_image(Image.open(p.name)) for p in ref_images]
        ref_tensor_cat = torch.concat(ref_tensors, axis=-1)

        c = {
            'lq_image': lq_tensor.unsqueeze(0).to(DEVICE),
            'ref_image': ref_tensor_cat.unsqueeze(0).to(DEVICE),
        }

        progress(0.2, desc="编码条件特征...")
        # Encode condition from image to latent
        with torch.no_grad():
            c = model.get_learned_conditioning(c)

        # CFG null condition = no reference
        uc = {k: c[k].detach().clone() for k in c.keys()}
        uc['ref_image'] *= 0

    except Exception as e:
        raise gr.Error(f"图片预处理失败: {e}")

    # Sample initial latent xT
    latent_shape = [model.model.diffusion_model.out_channels, model.image_size, model.image_size]
    torch.manual_seed(2024) # Use a fixed seed for reproducibility
    xT = torch.randn([1, *latent_shape], device=DEVICE)

    progress(0.4, desc="开始扩散去噪 (DDIM Sampling)...")
    # Diffusion denoising
    with torch.no_grad(), model.ema_scope():
        output_latent, _ = sampler.sample(
            S=50, # DDIM steps
            unconditional_guidance_scale=1.5, # CFG scale
            conditioning=c,
            unconditional_conditioning=uc,
            shape=latent_shape,
            x_T=xT,
            batch_size=1,
            verbose=False,
        )

    progress(0.9, desc="解码生成结果...")
    # Decode output latent to image
    with torch.no_grad():
        output_image = model.decode_first_stage(output_latent)

    # Post-process and save result
    output_image = ((output_image + 1.0) / 2.0).clamp(0.0, 1.0)
    output_image = output_image.squeeze(0).cpu()
    final_pil_image = to_pil_image(output_image)

    progress(1.0, desc="处理完成！")

    # Prepare comparison gallery
    comparison_gallery = [lq_image] + [Image.open(p.name) for p in ref_images] + [final_pil_image]
    
    return final_pil_image, comparison_gallery


# --- 3. Gradio UI Layout ---

def update_preview(lq_img, ref_list):
    """Updates the preview gallery when images are uploaded."""
    if lq_img is None and not ref_list:
        return None
    
    preview_list = []
    if lq_img:
        preview_list.append(lq_img)
    if ref_list:
        for ref_file in ref_list:
            try:
                preview_list.append(Image.open(ref_file.name))
            except Exception:
                # Handle cases where a file might not be a valid image
                pass
    return preview_list


css = """
.gradio-container { font-family: 'Microsoft YaHei', 'SimSun', sans-serif; }
.gr-button-primary { background: linear-gradient(to bottom right, #4A90E2, #0056b3); border-color: #0056b3; }
#preview_gallery .h-full { object-fit: cover; }
#result_gallery .h-full { object-fit: contain; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown("## Ref-LDM: 有参考的人脸图像修复")
    gr.Markdown("上传一张低质量（模糊、有噪点等）的人脸图片和 1-4 张高质量的参考人脸图片，模型将修复低质量图片。")

    with gr.Row(variant='panel'):
        # --- Left Column (Inputs) ---
        with gr.Column(scale=1):
            gr.Markdown("### 1. 输入控制")
            
            with gr.Accordion("上传图片", open=True):
                lq_input_img = gr.Image(
                    label="上传低质量图片 (LQ)", 
                    type="pil", 
                    image_mode="RGB"
                )
                ref_input_files = gr.File(
                    label="上传参考图片 (Ref, 最多4张)", 
                    file_count="multiple", 
                    file_types=["image"]
                )
            
            process_btn = gr.Button("开始处理", variant="primary")

        # --- Right Column (Outputs) ---
        with gr.Column(scale=2):
            gr.Markdown("### 2. 结果展示")
            
            with gr.Tab("输入预览"):
                preview_gallery = gr.Gallery(
                    label="输入图片预览", 
                    columns=5, 
                    height="auto", 
                    elem_id="preview_gallery"
                )
                gr.Markdown("上方第一张为低质量图，其余为参考图。")
            
            with gr.Tab("修复结果"):
                result_img = gr.Image(label="修复后图片", interactive=False)
                result_gallery = gr.Gallery(
                    label="效果对比 (LQ / Refs / Result)",
                    columns=5,
                    height="auto",
                    elem_id="result_gallery"
                )
                gr.Markdown("上方为最终修复结果，下方为修复前后及参考图的对比。")

    # --- 4. Event Listeners ---
    lq_input_img.upload(update_preview, inputs=[lq_input_img, ref_input_files], outputs=[preview_gallery])
    ref_input_files.upload(update_preview, inputs=[lq_input_img, ref_input_files], outputs=[preview_gallery])

    process_btn.click(
        fn=process_and_restore,
        inputs=[lq_input_img, ref_input_files],
        outputs=[result_img, result_gallery],
        api_name="restore_face"
    )


if __name__ == "__main__":
    # Check if model files exist before launching
    if not os.path.exists(MODEL_CKPT_PATH) or not os.path.exists(VAE_CKPT_PATH):
        print("\n[ERROR] 模型文件未找到！")
        print(f"请确保 '{MODEL_CKPT_PATH}' 和 '{VAE_CKPT_PATH}' 存在于项目中。")
        print("您可能需要运行 `download_ckpts.py` 或手动下载模型。")
    else:
        demo.queue().launch(share=True)