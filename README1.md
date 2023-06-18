# multilingual-dokugpt
Chat with multilingual documents in a language you prefer

`multilingual-dokugpt` is a quasi-localgpt that makes use of `langchain`, generates  embeddings `locally` and save them in a Chroma vectorstore.

## Features
* Handle documents in multiple languages, multiple formats (.txt or plaintext in other suffixes, .docx, .pdf, .epub)
* Can use PawanOsman reverse proxy free api-key of the form `pk-....`. Refer to `.env.sample`

## Usage
* Clone this repo
```
git clone https://github.com/ffreemt/multilingual-dokugpt
cd multilingual-dokugpt
```
* [Optional] Create a even
e.g.
```
python -m venv .venv
call .venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

* Install packages
```
python -m pip install -r requirements.txt
```

* Setup OPENAI_API_KEY
e.g.
```
set OPENAI_API_KEY=sk...  # Windows cmd
# export OPENAI_API_KEY=sk...  # bash

or setup .env (refer to `.env.sample`)
```

* Start the program
```
python main.py
```
* Browse to
```
http://127.0.0.1:7860
```

## TODO
* Make it a true localgpt by replacing the remote querying with a local `llm`.

## License
`multilingual-dokugpt` is released under the MIT License. See the LICENSE file for more details.
