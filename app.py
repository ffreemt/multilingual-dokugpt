"""Refer to
https://huggingface.co/spaces/mikeee/docs-chat/blob/main/app.py
and https://github.com/PromtEngineer/localGPT/blob/main/ingest.py

https://python.langchain.com/en/latest/getting_started/tutorials.html
"""
# pylint: disable=broad-exception-caught, unused-import, invalid-name, line-too-long, too-many-return-statements, import-outside-toplevel, no-name-in-module
import os
import time
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import gradio as gr
import torch
from charset_normalizer import detect
from chromadb.config import Settings
from epub2txt import epub2txt
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PDFMinerLoader,
    TextLoader,
)

# from constants import CHROMA_SETTINGS, SOURCE_DIRECTORY, PERSIST_DIRECTORY
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

# FAISS instead of PineCone
from langchain.vectorstores import FAISS, Chroma
from loguru import logger
from PyPDF2 import PdfReader  # localgpt
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline

# import click
# from typing import List

# from utils import xlxs_to_csv

# load possible env such as OPENAI_API_KEY
# from dotenv import load_dotenv

# load_dotenv()load_dotenv()

# fix timezone
os.environ["TZ"] = "Asia/Shanghai"
try:
    time.tzset()  # type: ignore # pylint: disable=no-member
except Exception:
    # Windows
    logger.warning("Windows, cant run time.tzset()")

ROOT_DIRECTORY = Path(__file__).parent
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/db"

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=PERSIST_DIRECTORY,
    anonymized_telemetry=False,
)
ns = SimpleNamespace(qa=None, ingest_done=None)


def load_single_document(file_path: str | Path) -> Document:
    """ingest.py"""
    # Loads a single document from a file path
    # encoding = detect(open(file_path, "rb").read()).get("encoding", "utf-8")
    encoding = detect(Path(file_path).read_bytes()).get("encoding", "utf-8")
    if file_path.endswith(".txt"):
        if encoding is None:
            logger.warning(
                f" {file_path}'s encoding is None "
                "Something is fishy, return empty str "
            )
            return Document(page_content="", metadata={"source": file_path})

        try:
            loader = TextLoader(file_path, encoding=encoding)
        except Exception as exc:
            logger.warning(f" {exc}, return dummy ")
            return Document(page_content="", metadata={"source": file_path})

    elif file_path.endswith(".pdf"):
        loader = PDFMinerLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)
    elif Path(file_path).suffix in [".docx"]:
        try:
            loader = Docx2txtLoader(file_path)
        except Exception as exc:
            logger.error(f" {file_path} errors: {exc}")
            return Document(page_content="", metadata={"source": file_path})
    elif Path(file_path).suffix in [".epub"]:  # for epub? epub2txt unstructured
        try:
            _ = epub2txt(file_path)
        except Exception as exc:
            logger.error(f" {file_path} errors: {exc}")
            return Document(page_content="", metadata={"source": file_path})
        return Document(page_content=_, metadata={"source": file_path})
    else:
        if encoding is None:
            logger.warning(
                f" {file_path}'s encoding is None "
                "Likely binary files, return empty str "
            )
            return Document(page_content="", metadata={"source": file_path})
        try:
            loader = TextLoader(file_path)
        except Exception as exc:
            logger.error(f" {exc}, returnning empty string")
            return Document(page_content="", metadata={"source": file_path})

    return loader.load()[0]


def get_pdf_text(pdf_docs):
    """docs-chat."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """docs-chat."""
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    """docs-chat."""
    # embeddings = OpenAIEmbeddings()
    model_name = "hkunlp/instructor-xl"
    model_name = "hkunlp/instructor-large"
    model_name = "hkunlp/instructor-base"
    logger.info(f"Loading {model_name}")
    embeddings = HuggingFaceInstructEmbeddings(model_name=model_name)
    logger.info(f"Done loading {model_name}")

    logger.info(
        "Doing vectorstore FAISS.from_texts(texts=text_chunks, embedding=embeddings)"
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    logger.info(
        "Done vectorstore FAISS.from_texts(texts=text_chunks, embedding=embeddings)"
    )

    return vectorstore


def greet(name):
    """Test."""
    logger.debug(f" name: [{name}] ")
    return "Hello " + name + "!!"


def upload_files(files):
    """Upload files."""
    file_paths = [file.name for file in files]
    logger.info(file_paths)

    ns.ingest_done = False
    res = ingest(file_paths)
    logger.info(f"Processed:\n{res}")

    # flag ns.qadone
    ns.ingest_done = True
    del res

    # ns.qa = load_qa()

    # return [str(elm) for elm in res]
    return file_paths

    # return ingest(file_paths)


def ingest(
    file_paths: list[str | Path], model_name="hkunlp/instructor-base", device_type=None
):
    """Gen Chroma db.

    torch.cuda.is_available()

    file_paths =
    ['C:\\Users\\User\\AppData\\Local\\Temp\\gradio\\41b53dd5f203b423f2dced44eaf56e72508b7bbe\\app.py',
    'C:\\Users\\User\\AppData\\Local\\Temp\\gradio\\9390755bb391abc530e71a3946a7b50d463ba0ef\\README.md',
    'C:\\Users\\User\\AppData\\Local\\Temp\\gradio\\3341f9a410a60ffa57bf4342f3018a3de689f729\\requirements.txt']
    """
    logger.info("\n\t Doing ingest...")

    if device_type is None:
        if torch.cuda.is_available():
            device_type = "cuda"
        else:
            device_type = "cpu"

    if device_type in ["cpu", "CPU"]:
        device = "cpu"
    elif device_type in ["mps", "MPS"]:
        device = "mps"
    else:
        device = "cuda"

    #  Load documents and split in chunks
    # logger.info(f"Loading documents from {SOURCE_DIRECTORY}")
    # documents = load_documents(SOURCE_DIRECTORY)

    documents = []
    for file_path in file_paths:
        documents.append(load_single_document(f"{file_path}"))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    logger.info(f"Loaded {len(documents)} documents ")
    logger.info(f"Split into {len(texts)} chunks of text")

    # Create embeddings
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=model_name, model_kwargs={"device": device}
    )

    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )
    db.persist()
    db = None
    logger.info("Done ingest")

    return [
        [Path(doc.metadata.get("source")).name, len(doc.page_content)]
        for doc in documents
    ]


# https://huggingface.co/TheBloke/vicuna-7B-1.1-HF
def gen_local_llm(model_id="TheBloke/vicuna-7B-1.1-HF"):
    """Gen a local llm.

    localgpt run_localgpt
    """
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    model = LlamaForCausalLM.from_pretrained(
        model_id,
        # load_in_8bit=True, # set these options if your GPU supports them!
        # device_map=1#'auto',
        # torch_dtype=torch.float16,
        # low_cpu_mem_usage=True
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm


def load_qa(device=None, model_name: str = "hkunlp/instructor-base"):
    """Gen qa."""
    logger.info("Doing qa")
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    # device = 'cpu'
    # model_name = "hkunlp/instructor-xl"
    # model_name = "hkunlp/instructor-large"
    # model_name = "hkunlp/instructor-base"
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=model_name, model_kwargs={"device": device}
    )
    # xl 4.96G, large 3.5G,
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )
    retriever = db.as_retriever()

    llm = gen_local_llm()  # "TheBloke/vicuna-7B-1.1-HF" 12G?

    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )

    logger.info("Done qa")

    return qa


def main1():
    """Lump codes"""
    with gr.Blocks() as demo:
        iface = gr.Interface(fn=greet, inputs="text", outputs="text")
        iface.launch()

    demo.launch()


def main():
    """Do blocks."""
    logger.info(f"ROOT_DIRECTORY: {ROOT_DIRECTORY}")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    logger.info(f"openai_api_key (env var/hf space SECRETS): {openai_api_key}")

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        # name = gr.Textbox(label="Name")
        # greet_btn = gr.Button("Submit")
        # output = gr.Textbox(label="Output Box")
        # greet_btn.click(fn=greet, inputs=name, outputs=output, api_name="greet")
        with gr.Accordion("Info", open=False):
            _ = """
                # localgpt
                Talk to your docs (.pdf, .docx, .epub, .txt .md and
                other text docs). It
                takes quite a while to ingest docs (10-30 min. depending
                on net, RAM, CPU etc.).
                """
            gr.Markdown(dedent(_))

        # with gr.Accordion("Upload files", open=True):
        with gr.Tab("Upload files"):
            # Upload files and generate embeddings database
            file_output = gr.File()
            upload_button = gr.UploadButton(
                "Click to upload files (Hold ctrl and click to select multiple files)",
                # file_types=["*.pdf", "*.epub", "*.docx"],
                file_count="multiple",
            )
            upload_button.upload(upload_files, upload_button, file_output)

        with gr.Tab("Query docs"):
            # interactive chat
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="Query")
            clear = gr.Button("Clear")

            def respond(message, chat_history):
                # bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
                if ns.ingest_done is None:  # no files processed yet
                    bot_message = "Upload some file(s) for processing first."
                    chat_history.append((message, bot_message))
                    return "", chat_history

                if not ns.ingest_done:  # embedding database not doen yet
                    bot_message = (
                        "Waiting for ingest (embedding) to finish, "
                        "be patient... You can switch the 'Upload files' "
                        "Tab to check"
                    )
                    chat_history.append((message, bot_message))
                    return "", chat_history

                if ns.qa is None:  # load qa one time
                    logger.info("Loading qa, need to do just one time.")
                    ns.qa = load_qa()

                try:
                    res = ns.qa(message)
                    answer, docs = res["result"], res["source_documents"]
                    bot_message = f"{answer} ({docs})"
                except Exception as exc:
                    logger.error(exc)
                    bot_message = f"bummer! {exc}"

                chat_history.append((message, bot_message))

                return "", chat_history

            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)

    try:
        from google import colab  # noqa

        share = True  # start share when in colab
    except Exception:
        share = False
    demo.launch(share=share)


if __name__ == "__main__":
    main()

_ = """
run_localgpt
device = 'cpu'
model_name = "hkunlp/instructor-xl"
model_name = "hkunlp/instructor-large"
model_name = "hkunlp/instructor-base"
embeddings = HuggingFaceInstructEmbeddings(
    model_name=,
    model_kwargs={"device": device}
)
# xl 4.96G, large 3.5G,
db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
retriever = db.as_retriever()

llm = gen_local_llm()  # "TheBloke/vicuna-7B-1.1-HF" 12G?

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

query = 'a'
res = qa(query)

"""
