"""Refer to https://huggingface.co/spaces/mikeee/docs-chat/blob/main/app.py.

and https://github.com/PromtEngineer/localGPT/blob/main/ingest.py

https://python.langchain.com/en/latest/getting_started/tutorials.html

gradio.Progress example:
    https://colab.research.google.com/github/gradio-app/gradio/blob/main/demo/progress/run.ipynb#scrollTo=2.8891853944186117e%2B38

unstructured: python-magic python-docx python-pptx
from langchain.document_loaders import UnstructuredHTMLLoader

docs = []
# for doc in Path('docs').glob("*.pdf"):
for doc in Path('docs').glob("*"):
# for doc in Path('docs').glob("*.txt"):
    docs.append(load_single_document(f"{doc}"))
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(docs)

model_name = "hkunlp/instructor-base"
embedding = HuggingFaceInstructEmbeddings(
    model_name=model_name, model_kwargs={"device": device}
)

# constitution.pdf 54344,   72 chunks Wall time: 3min 13s CPU times: total: 9min 4s @golay
# test.txt 21286,           27 chunks, Wall time: 47 s CPU times: total: 2min 30s @golay
# both                      99 chunks, Wall time: 5min 4s CPU times: total: 13min 31s
# chunks = len / 800

db = Chroma.from_documents(texts, embedding)

db = Chroma.from_documents(
    texts,
    embedding,
    persist_directory=PERSIST_DIRECTORY,
    client_settings=CHROMA_SETTINGS,
)
db.persist()
est. 1min/100 text1

# 中国共产党章程.txt qa
https://github.com/xanderma/Assistant-Attop/blob/master/Release/%E6%96%87%E5%AD%97%E7%89%88%E9%A2%98%E5%BA%93/31.%E4%B8%AD%E5%9B%BD%E5%85%B1%E4%BA%A7%E5%85%9A%E7%AB%A0%E7%A8%8B.txt

colab CPU test.text constitution.pdf
CPU times: user 1min 27s, sys: 8.09 s, total: 1min 35s
Wall time: 1min 37s

"""
# pylint: disable=broad-except, unused-import, invalid-name, line-too-long, too-many-return-statements, import-outside-toplevel, no-name-in-module, no-member, too-many-branches, unused-variable, too-many-arguments, global-statement
import os
import time
from copy import deepcopy
from math import ceil
from pathlib import Path

# from tempfile import _TemporaryFileWrapper
from textwrap import dedent
from types import SimpleNamespace
from typing import List

import gradio as gr
import httpx
import more_itertools as mit
import torch

# from about_time import about_time
from charset_normalizer import detect
from chromadb.config import Settings

# from langchain.embeddings import HuggingFaceInstructEmbeddings
# from langchain.llms import HuggingFacePipeline
# from epub2txt import epub2txt
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.docstore.document import Document
from langchain.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PDFMinerLoader,
    TextLoader,
)
from langchain.embeddings import (
    SentenceTransformerEmbeddings,
)  # HuggingFaceInstructEmbeddings,
from langchain.llms import HuggingFacePipeline, OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import (
    # CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.vectorstores import FAISS, Chroma
from loguru import logger
from PyPDF2 import PdfReader
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline

from epub_loader import EpubLoader
from load_api_key import load_api_key, pk_base, sk_base

# fix timezone
os.environ["TZ"] = "Asia/Shanghai"
try:
    time.tzset()  # type: ignore # pylint: disable=no-member
except Exception:
    # Windows
    logger.warning("Windows, cant run time.tzset()")

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

ROOT_DIRECTORY = Path(__file__).parent
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/db"

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=PERSIST_DIRECTORY,
    anonymized_telemetry=False,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"  # 1.11G

# opanai max 4097
# retriever default k = 4, query lenght about CHUNK_SIZE
# CHUNK_SIZE = about 4097 / 5: 820, with safety room: 625
CHUNK_SIZE = 625  # 250
CHUNK_OVERLAP = 60  # 50

ns_initial = SimpleNamespace(
    db=None,
    qa=None,
    texts=[],
    ingest_done=None,
    files_info=None,
    files_uploaded=[],
    db_ready=None,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    model_name=MODEL_NAME,
)
ns = deepcopy(ns_initial)


def load_single_document(file_path: str | Path) -> List[Document]:
    """Load a single document from a file path."""
    try:
        _ = Path(file_path).read_bytes()
        encoding = detect(_).get("encoding")
        if encoding is not None:
            encoding = str(encoding)
    except Exception as exc:
        logger.error(f"{file_path}: {exc}")
        encoding = None

    file_path = Path(file_path).as_posix()

    if Path(file_path).suffix in [".txt"]:
        if encoding is None:
            logger.warning(
                f" {file_path}'s encoding is None "
                "Something is fishy, return empty str "
            )
            return [Document(page_content="", metadata={"source": file_path})]
        try:
            loader = TextLoader(file_path, encoding=encoding)
        except Exception as exc:
            logger.warning(f" {exc}, return dummy ")
            return [Document(page_content="", metadata={"source": file_path})]
    elif Path(file_path).suffix in [".pdf"]:
        try:
            loader = PDFMinerLoader(file_path)
        except Exception as exc:
            logger.error(exc)
            return [Document(page_content="", metadata={"source": file_path})]
    elif file_path.endswith(".csv"):
        try:
            loader = CSVLoader(file_path)
        except Exception as exc:
            logger.error(exc)
            return [Document(page_content="", metadata={"source": file_path})]
    elif Path(file_path).suffix in [".docx"]:
        try:
            loader = Docx2txtLoader(file_path)
        except Exception as exc:
            logger.error(f" {file_path} errors: {exc}")
            return [Document(page_content="", metadata={"source": file_path})]
    elif Path(file_path).suffix in [".epub"]:
        try:
            # _ = epub2txt(file_path)
            loader = EpubLoader(file_path)
        except Exception as exc:
            logger.error(f" {file_path} errors: {exc}")
            return [Document(page_content="", metadata={"source": file_path})]
    else:
        if encoding is None:
            logger.warning(
                f" {file_path}'s encoding is None "
                "Likely binary files, return empty str "
            )
            return [Document(page_content="", metadata={"source": file_path})]
        try:
            loader = TextLoader(file_path)
        except Exception as exc:
            logger.error(f" {exc}, returnning empty string")
            return [Document(page_content="", metadata={"source": file_path})]

    return loader.load()  # use extend when combining


def get_pdf_text(pdf_docs):
    """docs-chat."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(f"{pdf}")  # taking care of Path
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# def get_text_chunks(text, chunk_size=None, chunk_overlap=None):
def get_doc_chunks(doc: Document, chunk_size=None, chunk_overlap=None) -> List[Document]:
    """Generate doc chunks."""
    if chunk_size is None:
        chunk_size = ns.chunk_size
    if chunk_overlap is None:
        chunk_overlap = ns.chunk_overlap

    # text_splitter = CharacterTextSplitter(
    text_splitter = RecursiveCharacterTextSplitter(
        # separator="\n",
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    # chunks = text_splitter.split_text(text)
    chunks = text_splitter.split_documents(doc)

    return chunks


def get_vectorstore(
    # text_chunks: List[Document],
    doc_chunks: List[Document],
    vectorstore=None,
    model_name=None,
    persist=True,
    persist_directory=None
):
    """Gne vectorstore."""
    # embedding = OpenAIEmbeddings()
    # for HuggingFaceInstructEmbeddings
    # model_name = "hkunlp/instructor-xl"
    # model_name = "hkunlp/instructor-large"
    # model_name = "hkunlp/instructor-base"

    # embedding = HuggingFaceInstructEmbeddings(model_name=model_name)

    if vectorstore is None:
        vectorstore = "chroma"

    if model_name is None:
        model_name = MODEL_NAME

    if persist_directory is None:
        persist_directory = PERSIST_DIRECTORY

    logger.info(f"Loading {model_name}")
    embedding = SentenceTransformerEmbeddings(model_name=model_name)
    logger.info(f"Done loading {model_name}")

    if vectorstore.lower() in ["chroma"]:
        logger.info(
            # "Doing vectorstore Chroma.from_texts(texts=text_chunks, embedding=embedding)"
            "Doing vectorstore Chroma.from_documents(texts=doc_chunks, embedding=embedding)"
        )
        if persist:
            # vectorstore = Chroma.from_texts(
            vectorstore = Chroma.from_documents(
                # texts=text_chunks,
                documents=doc_chunks,
                embedding=embedding,
                persist_directory=PERSIST_DIRECTORY,
                client_settings=CHROMA_SETTINGS,
            )
        else:
            # vectorstore = Chroma.from_texts(texts=text_chunks, embedding=embedding)
            vectorstore = Chroma.from_documents(documents=doc_chunks, embedding=embedding)

        logger.info(
            # "Done vectorstore Chroma.from_texts(texts=text_chunks, embedding=embedding)"
            "Done vectorstore Chroma.from_texts(documents=doc_chunks, embedding=embedding)"
        )

        return vectorstore

    # if vectorstore.lower() not in ['chroma']

    # TODO handle other cases

    logger.info(
        "Doing vectorstore FAISS.from_texts(documents=doc_chunks, embedding=embedding)"
    )
    # vectorstore = FAISS.from_texts(documents=doc_chunks, embedding=embedding)
    vectorstore = FAISS.from_documents(documents=doc_chunks, embedding=embedding)
    logger.info(
        "Done vectorstore FAISS.from_documents(documents=doc_chunks, embedding=embedding)"
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

    ns.files_uploaded = file_paths

    # return [str(elm) for elm in res]
    return file_paths

    # return ingest(file_paths)


def process_files(
    # file_paths,
    progress=gr.Progress(),
):
    """Process uploaded files."""
    if not ns.files_uploaded:
        return f"No files uploaded: {ns.files_uploaded}"

    # wait for update before querying new ns.qa
    ns.ingest_done = False

    logger.debug(f"ns.files_uploaded: {ns.files_uploaded}")

    # imgs = [None] * 24
    # for img in progress.tqdm(imgs, desc="Loading from list"):
    # time.sleep(0.1)

    # imgs = [[None] * 8] * 3
    # for img_set in progress.tqdm(imgs, desc="Nested list"):
    # time.sleep(.2)
    # for img in progress.tqdm(img_set, desc="inner list"):
    # time.sleep(10.1)

    # return "done..."

    documents = []
    if progress is None:
        for file_path in ns.files_uploaded:
            logger.debug(f"-Doing {file_path}")
            try:
                documents.extend(load_single_document(f"{file_path}"))
                logger.debug("-Done reading files.")
            except Exception as exc:
                logger.error(f"-{file_path}: {exc}")
    else:
        for file_path in progress.tqdm(ns.files_uploaded, desc="Reading file(s)"):
            logger.debug(f"Doing {file_path}")
            try:
                documents.extend(load_single_document(f"{file_path}"))
                logger.debug("Done reading files.")
            except Exception as exc:
                logger.error(f"{file_path}: {exc}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=ns.chunk_size, chunk_overlap=ns.chunk_overlap
    )
    texts = text_splitter.split_documents(documents)

    logger.info(f"Loaded {len(ns.files_uploaded)} files ")
    logger.info(f"Loaded {len(documents)} document(s) ")
    logger.info(f"Split into {len(texts)} chunk(s) of text")

    total = ceil(len(texts) / 101)
    ns.texts = texts

    ns.ingest_done = True
    _ = [
        [Path(doc.metadata.get("source")).name, len(doc.page_content)]
        for doc in documents
    ]
    ns.files_info = _

    _ = (
        f"done file(s): {dict(ns.files_info)}, split to "
        f"{total} chunk(s). \n\nThe following embedding takes "
        f" {total} step(s). (Each step lasts about 18 secs "
        " on a free tier instance on huggingface space.)"
    )

    return _


def embed_files(progress=gr.Progress()):
    """Embded ns.files_uploaded."""
    # initialize if necessary
    if ns.db is None:
        logger.info(f"loading {ns.model_name:}")
        embedding = SentenceTransformerEmbeddings(
            model_name=ns.model_name, model_kwargs={"device": DEVICE}
        )
        for _ in progress.tqdm(range(1), desc="diggin..."):
            logger.info("creating vectorstore")
            ns.db = Chroma(
                # persist_directory=PERSIST_DIRECTORY,
                embedding_function=embedding,
                # client_settings=CHROMA_SETTINGS,
            )
        logger.info("done creating vectorstore")

    total = ceil(len(ns.texts) / 101)
    if progress is None:
        # for text in progress.tqdm(
        for idx, text in enumerate(mit.chunked_even(ns.texts, 101)):
            logger.debug(f"-{idx + 1} of {total}")
            ns.db.add_documents(documents=text)
    else:
        # for text in progress.tqdm(
        for idx, text in enumerate(
            progress.tqdm(
                mit.chunked_even(ns.texts, 101),
                total=total,
                desc="Processing docs",
            )
        ):
            logger.debug(f"{idx + 1} of {total}")
            ns.db.add_documents(documents=text)
    logger.debug(f" done all {total}")

    # ns.qa = load_qa()

    llm = OpenAI(temperature=0, max_tokens=1024)  # type: ignore
    retriever = ns.db.as_retriever()
    ns.qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        # return_source_documents=True,
    )

    logger.debug(f"{ns.ingest_done=}, exit process_files")

    _ = (
        f"Done {total} chunk(s). Now "
        "switch to Query Docs Tab to chat. "
        "You can chat in a language you prefer, "
        "independent of the document language. Have fun."
    )
    return _


def respond(message, chat_history):
    """Gen response."""
    logger.debug(f"{ns.files_uploaded=}")
    if not ns.files_uploaded:  # no files processed yet
        bot_message = "Upload some file(s) for processing first."
        chat_history.append((message, bot_message))
        return "", chat_history

    logger.debug(f"{ns.ingest_done=}")
    if not ns.ingest_done:  # embedding database not doen yet
        bot_message = (
            "Waiting for ingest (embedding) to finish, "
            "be patient... You can switch the 'Upload files' "
            "Tab to check"
        )
        chat_history.append((message, bot_message))
        return "", chat_history

    _ = """
    if ns.qa is None:  # load qa one time
        logger.info("Loading qa, need to do just one time.")
        ns.qa = load_qa()
        logger.info("Done loading qa, need to do just one time.")
    # """
    if ns.qa is None:
        bot_message = "Looks like the bot is not ready. Try again later..."
        chat_history.append((message, bot_message))
        return "", chat_history

    try:
        res = ns.qa(message)
        answer = res.get("result")
        docs = res.get("source_documents")
        if docs:
            bot_message = f"{answer}\n({docs})"
        else:
            bot_message = f"{answer}"
    except Exception as exc:
        logger.error(exc)
        bot_message = f"bummer! {exc}"
        if "empty" in str(exc):
            bot_message = f"{bot_message} (probably invalid apikey)"

    chat_history.append((message, bot_message))

    return "", chat_history


# pylint disable=unused-argument
def ingest(
    file_paths: list[str | Path],
    model_name: str = MODEL_NAME,
    device_type=None,
    chunk_size: int = 256,
    chunk_overlap: int = 50,
):
    """Gen Chroma db."""
    logger.info("\n\t Doing ingest...")
    logger.debug(f" file_paths: {file_paths}")
    logger.debug(f"type of file_paths: {type(file_paths)}")

    # raise SystemExit(0)

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
        # documents.append(load_single_document(f"{file_path}"))
        logger.debug(f"Doing {file_path}")
        documents.extend(load_single_document(f"{file_path}"))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)

    logger.info(f"Loaded {len(file_paths)} files ")
    logger.info(f"Loaded {len(documents)} documents ")
    logger.info(f"Split into {len(texts)} chunks of text")

    # Create embedding
    # embedding = HuggingFaceInstructEmbeddings(
    embedding = SentenceTransformerEmbeddings(
        model_name=model_name, model_kwargs={"device": device}
    )

    # https://stackoverflow.com/questions/76048941/how-to-combine-two-chroma-databases
    # db = Chroma(persist_directory=chroma_directory, embedding_function=embedding)
    # db.add_documents(documents=texts1)

    # mit.chunked_even(texts, 100)
    db = Chroma(
        # persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding,
        # client_settings=CHROMA_SETTINGS,
    )
    # for text in progress.tqdm(
    for text in tqdm(mit.chunked_even(texts, 101), total=ceil(len(texts) / 101)):
        db.add_documents(documents=text)

    _ = """
    with about_time() as atime:  # type: ignore
        db = Chroma.from_documents(
            texts,
            embedding,
            persist_directory=PERSIST_DIRECTORY,
            client_settings=CHROMA_SETTINGS,
        )
    logger.info(f"Time spent: {atime.duration_human}")  # type: ignore
    """

    logger.info(f"persist_directory: {PERSIST_DIRECTORY}")

    # db.persist()
    # db = None
    # ns.db = db
    ns.qa = db

    logger.info("Done ingest")

    _ = [
        [Path(doc.metadata.get("source")).name, len(doc.page_content)]
        for doc in documents
    ]
    ns.files_info = _

    return _


# TheBloke/Wizard-Vicuna-7B-Uncensored-HF
# https://huggingface.co/TheBloke/vicuna-7B-1.1-HF
def gen_local_llm(model_id="TheBloke/vicuna-7B-1.1-HF"):
    """Gen a local llm.

    localgpt run_localgpt
    https://medium.com/pytorch/bettertransformer-out-of-the-box-performance-for-huggingface-transformers-3fbe27d50ab2
    with torch.device(“cuda”):
        model = AutoModelForCausalLM.from_pretrained(“gpt2-large”, torch_dtype=torch.float16)

        model = BetterTransformer.transform(model)
    """
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    if torch.cuda.is_available():
        model = LlamaForCausalLM.from_pretrained(
            model_id,
            # load_in_8bit=True, # set these options if your GPU supports them!
            # device_map=1  # "auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(model_id)

    local_llm = None
    if model is not None:  # to please pyright
        pipe = pipeline(
            "text-generation",
            model=model,  # type: ignore
            tokenizer=tokenizer,
            max_length=2048,
            temperature=0,
            top_p=0.95,
            repetition_penalty=1.15,
        )

        local_llm = HuggingFacePipeline(pipeline=pipe)

    return local_llm


def load_qa(device=None, model_name: str = MODEL_NAME):
    """Gen qa.

    device = 'cpu'
    model_name = "hkunlp/instructor-xl"
    model_name = "hkunlp/instructor-large"
    model_name = "hkunlp/instructor-base"
    embedding = HuggingFaceInstructEmbeddings(
    """
    logger.info("Doing qa")
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    embedding = SentenceTransformerEmbeddings(
        model_name=model_name, model_kwargs={"device": device}
    )
    # xl 4.96G, large 3.5G,

    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding,
        client_settings=CHROMA_SETTINGS,
    )
    retriever = db.as_retriever()

    # _ = """
    # llm = gen_local_llm()  # "TheBloke/vicuna-7B-1.1-HF" 12G?

    llm = OpenAI(temperature=0, max_tokens=1024)  # type: ignore
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        # return_source_documents=True,
    )

    # {"query": ..., "result": ..., "source_documents": ...}

    return qa

    # TODO: conversation_chain
    # pylint: disable=unreachable

    # model = 'gpt-3.5-turbo', default text-davinci-003
    # max_tokens: int = 256 max_retries: int = 6
    # openai_api_key: Optional[str] = None,
    # openai_api_base: Optional[str] = None,

    # llm = OpenAI(temperature=0, max_tokens=0)
    llm = OpenAI(temperature=0, max_tokens=1024)  # type: ignore
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        # retriever=vectorstore.as_retriever(),
        retriever=db.as_retriever(),
        memory=memory,
    )

    logger.info("Done qa")

    return conversation_chain
    # memory.clear()
    # response = conversation_chain({'question': user_question})
    # response['question'], response['answer']


def main1():
    """Lump codes."""
    with gr.Blocks() as demo1:
        iface = gr.Interface(fn=greet, inputs="text", outputs="text")
        iface.launch()

    demo1.launch()


logger.info(f"ROOT_DIRECTORY: {ROOT_DIRECTORY}")

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE")
logger.info(f"openai_api_key (env var/hf space SECRETS): {openai_api_key}")
logger.info(f"openai_api_base: {openai_api_base}")

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # name = gr.Textbox(label="Name")
    # greet_btn = gr.Button("Submit")
    # output = gr.Textbox(label="Output Box")
    # greet_btn.click(fn=greet, inputs=name, outputs=output, api_name="greet")
    #
    #  ### layout ###
    with gr.Accordion("Info", open=False):
        _ = """
            # localgpt
            Talk to your docs (.pdf, .docx, .epub, .txt .md and
            other text docs). It
            takes quite a while to ingest docs (10-30 min. depending
            on net, RAM, CPU etc.).

            Send empty query (hit Enter) to check embedding status and files info ([filename, numb of chars])

            Homepage: https://huggingface.co/spaces/mikeee/localgpt
            """
        gr.Markdown(dedent(_))

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

        # reset_btn = gr.Button("Reset everything", visibile=False)

    with gr.Tab("Query docs"):
        # interactive chat
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Query")
        clear = gr.Button("Clear")

    # actions
    def reset_all():
        """Reset ns."""
        global ns
        ns = deepcopy(ns_initial)
        return f"reset done: ns={ns}"

    # reset_btn.click(reset_all, [], text2)

    upload_button.upload(upload_files, upload_button, file_output)
    process_btn.click(process_files, [], text2)

    # Query docs TAB
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.queue(concurrency_count=20).launch()

_ = """
run_localgpt
device = 'cpu'
model_name = "hkunlp/instructor-xl"
model_name = "hkunlp/instructor-large"
model_name = "hkunlp/instructor-base"
embedding = HuggingFaceInstructEmbeddings(
    model_name=,
    model_kwargs={"device": device}
)
# xl 4.96G, large 3.5G,
db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding, client_settings=CHROMA_SETTINGS)
retriever = db.as_retriever()

llm = gen_local_llm()  # "TheBloke/vicuna-7B-1.1-HF" 12G?

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

query = 'a'
res = qa(query)

---
https://www.linkedin.com/pulse/build-qa-bot-over-private-data-openai-langchain-leo-wang

history = [】

def user(user_message, history):
    # Get response from QA chain
    response = qa({"question": user_message, "chat_history": history})
    # Append user message and response to chat history
    history.append((user_message, response["answer"]))]

---
https://llamahub.ai/l/file-unstructured

from pathlib import Path
from llama_index import download_loader

UnstructuredReader = download_loader("UnstructuredReader")

loader = UnstructuredReader()
documents = loader.load_data(file=Path('./10k_filing.html'))

# --
from pathlib import Path
from llama_index import download_loader

# SimpleDirectoryReader = download_loader("SimpleDirectoryReader")
# FileNotFoundError: [Errno 2] No such file or directory

documents = SimpleDirectoryReader('./data').load_data()

loader = SimpleDirectoryReader('./data', file_extractor={
  ".pdf": "UnstructuredReader",
  ".html": "UnstructuredReader",
  ".eml": "UnstructuredReader",
  ".pptx": "PptxReader"
})
documents = loader.load_data()
"""
