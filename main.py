"""Test."""
# pylint: disable=invalid-name, unused-import, broad-except,
import os
from copy import deepcopy
from textwrap import dedent

import gradio as gr
import httpx
from loguru import logger

from app import (embed_files, ingest, ns, ns_initial, process_files, respond,
                 upload_files)
from load_api_key import load_api_key, pk_base, sk_base

api_key = load_api_key()
if api_key is not None:
    os.environ.setdefault("OPENAI_API_KEY", api_key)
    if api_key.startswith("sk-"):
        os.environ.setdefault("OPENAI_API_BASE", sk_base)
    elif api_key.startswith("pk-"):
        os.environ.setdefault("OPENAI_API_BASE", pk_base)
        # resetip
        try:
            url = "https://api.pawan.krd/resetip"
            headers = {"Authorization": f"{api_key}"}
            httpx.post(url, headers=headers)
        except Exception as exc_:
            logger.error(exc_)
            raise

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE")
logger.info(f"openai_api_key (env var/hf space SECRETS): {openai_api_key}")
logger.info(f"openai_api_base: {openai_api_base}")

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Tab("📚 Upload files"):  # Tab1
        with gr.Accordion("🎈 Info", open=False):
            _ = """
                ### multilingual dokugpt/多语dokugpt

                和你的文件对话： 可用中文向外语文件提问或用外语向中文文件提问

                Talk to your docs (.pdf, .docx, .epub, .txt .md and
                other text docs): You can ask questions in a language you prefer, independent of the document language.

                It
                takes quite a while to ingest docs (5-30 min. depending
                on net, RAM, CPU etc.).

                Send empty query (hit Enter) to check embedding status and files info ([filename, numb of chars])

                Homepage: https://huggingface.co/spaces/mikeee/multilingual-dokugpt
                """
            gr.Markdown(dedent(_))

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
            text2 = gr.Textbox("Process docs")
            process_btn = gr.Button("Click to process")
        with gr.Row():
            text_embed = gr.Textbox("Generate embeddings")
            embed_btn = gr.Button("Click to embed")

        reset_btn = gr.Button("Reset everything", visible=True)

    with gr.Tab("🤖 Query docs"):  # Tab1
        # interactive chat
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Query")
        with gr.Row():
            msg_submitbtn = gr.Button("Submit")
            clear = gr.Button("Clear")

    # actions
    def reset_all():
        """Reset ns."""
        # global ns
        globals().update(**{"ns": deepcopy(ns_initial)})
        logger.debug(f"reset to {ns=}")
        return f"reset done: ns={ns}"

    # Tab1
    upload_button.upload(upload_files, upload_button, file_output)
    process_btn.click(process_files, [], text2)
    embed_btn.click(embed_files, [], text_embed)
    reset_btn.click(reset_all, [], text2)

    # Tab2
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    msg_submitbtn.click(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=True)


if __name__ == "__main__":
    demo.queue(concurrency_count=20).launch()
