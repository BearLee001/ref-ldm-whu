import gradio as gr

from ldm.app.callbacks import process_and_restore, update_preview
from ldm.app.UI import CSS, create_ui_layout

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
    demo.queue().launch(share=True)