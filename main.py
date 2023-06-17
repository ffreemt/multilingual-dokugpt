"""Test."""
# pylint: disable=invalid-name, unused-import, broad-except,
from copy import deepcopy

import gradio as gr
from app import ingest, ns, ns_initial, process_files, upload_files, respond
from load_api_key import load_api_key, pk_base, sk_base
from loguru import logger

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Tab("Upload files"):
        # Upload files and generate vectorstore
        with gr.Row():
            file_output = gr.File()
            # file_output = gr.Text()
            # file_output = gr.DataFrame()
            upload_button = gr.UploadButton(
                "Click to upload",
                # file_types=["*.pdf", "*.epub", "*.docx"],
                file_count="multiple",
            )
        with gr.Row():
            text2 = gr.Textbox("Gen embedding")
            process_btn = gr.Button("Click to embed")

        reset_btn = gr.Button("Reset everything", visible=False)

    with gr.Tab("Query docs"):
        # interactive chat
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Query")
        clear = gr.Button("Clear")

    # actions
    def reset_all():
        """Reset ns."""
        # global ns
        globals().update(**{"ns": deepcopy(ns_initial)})
        return f"reset done: ns={ns}"

    reset_btn.click(reset_all, [], text2)

    upload_button.upload(upload_files, upload_button, file_output)
    process_btn.click(process_files, [], text2)

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.queue(concurrency_count=20).launch()
