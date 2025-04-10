# Setting up to use Python retrieval-augmented-generation

## On MacOS

1. Install brew package manager, makes working with different Python versions easy
Open Terminal
```
john@localhost:~ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
```
john@localhost:~ brew install python@3.10
```
Choose where you want the project to live, can be moved later, I've placed mine in my repos folder
```
john@localhost:~ mkdir repos
john@localhost:~ cd repos
john@localhost repos% python3.10 -m venv repos/rag_env
```

2. Clone my repo with python scripts and requirements.txt
```
john@localhost repos% git clone git@github.com:jebucha/rag_llm.git
```

3. Install Ollama for MacOS and pull 2 models to start with
https://ollama.com/download/Ollama-darwin.zip
Unzip that and copy / move Ollama.app to your Applicatins folder, then run it
After Ollama is up and running, back to Terminal

```
john@localhost:~ ollama pull mistral
john@localhost:~ ollama pull cogito
```

4. You can rename the folder if you like, but just to get you started
```
john@localhost repos% cd rag_llm
john@localhost rag_llm% source ../rag_env/bin/activate
(rag_env) john@localhost rag_llm% pip install -r requirements.txt
(rag_env) john@localhost rag_llm% python ingest_multiple_md.py
[ This will prompt you for the path to your Markdown files, can be relative or absolute, will then ingest and process them ]
(rag_llm) john@localhost rag_llm% python verify_count.py
(rag_llm) john@localhost rag_llm% python ask_md.py
```

[ This will prompt you for the question you wish to ask, it then transforms or "embeds" your question into vector representation, performs search against vector db finding similar sentences or documents, builds LLM prompt using the retrieved data as an instruction "Use the following documents to answer your question", and then the question itself. I've added 3 print lines to the ask_md.py so I could see what was being sent to the LLM. ]
 


## Basic setup instructions on Debian/Ubuntu/Pop!_OS

1. 
```
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral
ollama pull cogito
```
In the ask_md.py, towards the bottom, try alternating between the mistral and cogito models.

2.
```
mkdir repos
cd repos
git clone git@github.com:jebucha/rag_llm.git
apt install -y python3.10-venv
```
3.
```
$(homedir) python3.10 -m venv rag_env
$(homedir) cd rag_env
$(homedir) source bin/activate
(rag_env) john@pop-os:~/repos/rag_env$ cp ../rag_llm/*.py .
(rag_env) john@pop-os:~/repos/rag_env$ cp ../rag_llm/requirements.txt .
(rag_env) john@pop-os:~/repos/rag_env$ pip install -r requirements.txt
```

4.
Edit the ingest_multiple_md.py script, just down from the top you'll see the block below, change the path to where your markdown files are located, save & quit.
```
# Directory containing Markdown files
md_directory = "/home/john/repos/lists"  # Adjust this to your directory path
persist_path = "./chroma_db"
```
5.
```
(rag_env) john@pop-os:~/repos/rag_env$ python ./ingest_multiple_md.py
```
Output should show something like:

Ingested 19 chunks from (path to first md file processed)
Ingested 15 chunks from (path to second md file processed)
...
Finished ingesting 165 chunks from 25 files.

6. 
```
(rag_env) john@pop-os:~/repos/rag_env$ python ./ask_md.py
```

You'll be prompted to enter a question (natural language), it may take a bit depending on the performance of your computer, download speed of the Mistral model from Ollama (it's about 4.1GB), etc but you it should return an answer drawn from your documents.

This isn't necessarily a precise word for word but when input files are "embedded" / transformed the process is roughly as follows:

* Text Embeddings: Generate dense vector representations of sentences, paragraphs, or documents.
* Semantic Search: Find similar sentences or documents based on their semantic meaning.
* Text Classification: Use sentence embeddings as input features for text classification tasks.
* Clustering: Group similar sentences or documents together based on their embeddings.
* Information Retrieval: Improve search results by using sentence embeddings to rank documents.

7. For the rag_ui.py, that launches a simple web ui locally using a Python library called "streamlit". To use that you should be able to run
```
(rag_env) john@pop-os:~/repos/rag_env$ streamlit run rag_ui.py
```

That should launch your default browser and load http://localhost:8501, and present a simple input to run queries against your processed data.
