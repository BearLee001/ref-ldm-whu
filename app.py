import gradio as gr
import os

from ldm.app.callbacks import process_and_restore, update_preview
from ldm.app.UI import CSS, create_ui_layout
from ldm.global_config import VAE_CKPT_PATH, MODEL_CKPT_PATH
with gr.Blocks(theme=gr.themes.Soft(), css=CSS) as demo:
    ui = create_ui_layout()
    ui.lq_input_img.upload(
        fn=update_preview,
        inputs=[ui.lq_input_img, ui.ref_input_files],
        outputs=[ui.preview_gallery]
    )
    ui.ref_input_files.upload(
        fn=update_preview,
        inputs=[ui.lq_input_img, ui.ref_input_files],
        outputs=[ui.preview_gallery]
    )
    ui.process_btn.click(
        fn=process_and_restore,
        inputs=[ui.lq_input_img, ui.ref_input_files],
        outputs=[ui.result_img, ui.result_gallery],
        api_name="restore_face"
    )

if __name__ == "__main__":
    if not os.path.exists(MODEL_CKPT_PATH) or not os.path.exists(VAE_CKPT_PATH):
        print("\n[ERROR] 模型文件未找到！")
        print(f"请确保 '{MODEL_CKPT_PATH}' 和 '{VAE_CKPT_PATH}' 存在于项目中。")
        print("您可能需要运行 `download_ckpts.py` 或手动下载模型。")
    else:
        demo.queue().launch(share=True)