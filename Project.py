# -*- coding: utf-8 -*-
"""
Math Problem Retrieval System (RAG + Embeddings)

A Python project that builds a semantic search and code generation pipeline
for mathematical problems using vector embeddings, FAISS, ChromaDB,
and a RAG chain powered by OpenAI.

Originally developed in Google Colab.
"""

import re
import time
import os
import ast

import pandas as pd
import numpy as np
import faiss
from datasets import load_dataset, Dataset, Features, Value
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


# ============================================================
# STEP 1: Dataset Cleaning & Deduplication
# ============================================================

def curata_problem_statement(text):
    """Curatarea textului unei probleme (care vine din coloana 'prompt')."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.replace('$', '')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    if text.endswith(('.', '?')):
        text = text[:-1].strip()
    return text


def curata_python_solution(text):
    """Curatarea solutiei Python (care vine din coloana 'completion')."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    return text


def get_unique_indices(data_list):
    """Returneaza indicii primelor aparitii ale elementelor unice dintr-o lista."""
    seen = set()
    indices = []
    for idx, item in enumerate(data_list):
        if item not in seen:
            seen.add(item)
            indices.append(idx)
    return indices


print("1. Incarcare set de date")
start_time = time.time()
dataset = None
try:
    dataset = load_dataset("sdiazlor/math-python-reasoning-dataset", split="train")
    print("   Incarcat cu succes split='train'.")
except Exception as e:
    print(f"   Nu am putut incarca split='train', incerc fara split specificat. Eroare: {e}")
    try:
        dataset = load_dataset("sdiazlor/math-python-reasoning-dataset")
        if isinstance(dataset, dict) and "train" in dataset:
            print("   Dataset incarcat ca dict, selectez split='train'.")
            dataset = dataset['train']
        elif isinstance(dataset, dict):
            print(f"   Dataset incarcat ca dict, dar nu contine split='train'. Splituri disponibile: {list(dataset.keys())}")
            dataset = None
        else:
            print("   Dataset incarcat cu succes (probabil default='train').")
    except Exception as e2:
        print(f"   Nu am putut incarca setul de date. Eroare finala: {e2}")

if dataset is None:
    print("\n!!! EROARE FATALA: Nu s-a putut incarca setul de date.")
    exit()

load_time = time.time() - start_time
print(f"   Set de date incarcat in {load_time:.2f} secunde.")
print(f"   Numar initial de exemple: {len(dataset)}")
print(f"   Coloane detectate: {dataset.column_names}")
print(f"   Exemplu date originale:\n{dataset[0]}\n")

print("2. Curatare date (prompt -> problem_cleaned, completion -> solution_cleaned)...")
start_time = time.time()
dataset_curatat = dataset.map(
    lambda exemplu: {
        'problem_cleaned': curata_problem_statement(exemplu['prompt']),
        'solution_cleaned': curata_python_solution(exemplu['completion'])
    },
    num_proc=4,
    remove_columns=dataset.column_names
)
map_time = time.time() - start_time
print(f"   Curatare aplicata in {map_time:.2f} secunde.")
print(f"   Coloane dupa curatare: {dataset_curatat.column_names}")
print(f"   Exemplu date dupa curatare:\n{dataset_curatat[0]}\n")

print("3. Identificare si eliminare duplicate bazate pe 'problem_cleaned'...")
start_time = time.time()
probleme_curatate_lista = dataset_curatat['problem_cleaned']
indici_unici = get_unique_indices(probleme_curatate_lista)
num_duplicates = len(dataset_curatat) - len(indici_unici)
dedup_time = time.time() - start_time
print(f"   Identificare indici unici in {dedup_time:.2f} secunde.")
print(f"   Numar de duplicate eliminate: {num_duplicates}")

print("4. Creare set de date final deduplicat...")
start_time = time.time()
dataset_final = dataset_curatat.select(indici_unici)
select_time = time.time() - start_time
print(f"   Selectare randuri unice in {select_time:.2f} secunde.")
print(f"   Numar final de exemple: {len(dataset_final)}")

print(f"\n--- Procesare completa ---")
print(f"Numar initial de probleme: {len(dataset)}")
print(f"Numar final de probleme unice: {len(dataset_final)}")

if len(dataset_final) > 4:
    print(f"\nExemplu din setul de date final:\n{dataset_final[4]}")
elif len(dataset_final) > 0:
    print(f"\nExemplu din setul de date final:\n{dataset_final[0]}")

output_csv_filename = "math_python_dataset_curatat.csv"
print(f"\nSalvare set de date curatat ca fisier CSV: '{output_csv_filename}'...")
try:
    dataset_final.to_csv(output_csv_filename, index=False, encoding='utf-8')
    print(f"   Set de date salvat cu succes ca '{output_csv_filename}'.")
except Exception as e:
    print(f"   EROARE la salvarea fisierului CSV: {e}")


# ============================================================
# STEP 2: Embedding Generation & Model Comparison (FAISS)
# ============================================================

MODELE_DE_TESTAT = ['all-MiniLM-L6-v2', 'BAAI/bge-base-en-v1.5']
K_SEARCH = 5

print(f"\n{'='*60}")
print("STEP 2: Embedding Generation & Model Comparison")
print(f"{'='*60}")

print(f"\n1. Incarcare probleme din '{output_csv_filename}'...")
try:
    df = pd.read_csv(output_csv_filename)
    lista_probleme_curatate = df['problem_cleaned'].fillna("").astype(str).tolist()
    if not lista_probleme_curatate:
        raise ValueError("Lista de probleme este goala dupa incarcare.")
    print(f"   Am incarcat {len(lista_probleme_curatate)} probleme.")
except Exception as e:
    print(f"EROARE la citirea CSV-ului: {e}")
    exit()

print("\n2. Definire set de referinta...")
set_referinta = [
    {"query_index": 10, "query_text": "what is the sum of 5 and 7", "expected_indices": [25, 150]},
    {"query_index": 45, "query_text": "multiply 6 by 3", "expected_indices": [90]},
    {"query_index": 82, "query_text": "if x = 10 and y = 4 what is x - y", "expected_indices": [120, 30]},
    {"query_index": 15, "query_text": "calculate area of rectangle length 8 width 2", "expected_indices": [200]},
]
print(f"   Set de referinta definit cu {len(set_referinta)} intrari.")

print(f"\n3. Testare modele de embedding (k={K_SEARCH})...")
results = {}

for model_name in MODELE_DE_TESTAT:
    print(f"\n--- Testare Model: {model_name} ---")
    results[model_name] = {
        "encoding_time": None,
        "avg_search_time": None,
        "hits_at_k": 0,
        "total_queries": len(set_referinta),
        "precision_at_k_sum": 0.0,
        "recall_at_k_sum": 0.0
    }

    try:
        print("   Incarcare model...")
        start_load = time.time()
        model = SentenceTransformer(model_name)
        print(f"   Model incarcat in {time.time() - start_load:.2f} sec.")

        print(f"   Generare embeddings pentru {len(lista_probleme_curatate)} probleme...")
        start_encode = time.time()
        embeddings = model.encode(lista_probleme_curatate, show_progress_bar=True, normalize_embeddings=True)
        encoding_time = time.time() - start_encode
        results[model_name]["encoding_time"] = encoding_time
        print(f"   Embeddings generate in {encoding_time:.2f} sec.")

        embedding_dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(embedding_dim)
        index.add(np.array(embeddings).astype('float32'))
        print(f"   Index FAISS creat (dim={embedding_dim}, {index.ntotal} vectori).")

        print("   Rulare cautari pe setul de referinta...")
        total_search_time = 0
        num_searches = 0

        for item_ref in set_referinta:
            query_idx = item_ref["query_index"]

            if query_idx < 0 or query_idx >= len(lista_probleme_curatate):
                print(f"   Avertisment: query_index {query_idx} invalid. Omitere.")
                results[model_name]["total_queries"] -= 1
                continue

            query_text = lista_probleme_curatate[query_idx]
            expected_indices = set(item_ref["expected_indices"])
            num_expected = len(expected_indices)

            start_search_single = time.time()
            query_embedding = model.encode([query_text], normalize_embeddings=True)
            distances, retrieved_indices_with_dist = index.search(
                np.array(query_embedding).astype('float32'), K_SEARCH
            )
            total_search_time += (time.time() - start_search_single)
            num_searches += 1

            retrieved_indices = set(retrieved_indices_with_dist[0])
            retrieved_indices_no_self = retrieved_indices - {query_idx}
            relevant_found = retrieved_indices_no_self.intersection(expected_indices)
            num_relevant_found = len(relevant_found)

            if num_relevant_found > 0:
                results[model_name]["hits_at_k"] += 1

            precision_k = num_relevant_found / K_SEARCH if K_SEARCH > 0 else 0.0
            results[model_name]["precision_at_k_sum"] += precision_k

            recall_k = num_relevant_found / num_expected if num_expected > 0 else 0.0
            results[model_name]["recall_at_k_sum"] += recall_k

        if num_searches > 0:
            results[model_name]["avg_search_time"] = total_search_time / num_searches
            avg_precision = results[model_name]["precision_at_k_sum"] / num_searches
            avg_recall = results[model_name]["recall_at_k_sum"] / num_searches
            hit_rate = results[model_name]["hits_at_k"] / num_searches

            print(f"   Evaluare finalizata pentru {model_name}:")
            print(f"      Timp mediu cautare: {results[model_name]['avg_search_time']:.4f} sec")
            print(f"      Hit Rate@{K_SEARCH}:   {hit_rate:.2%}")
            print(f"      Precision@{K_SEARCH}:  {avg_precision:.4f}")
            print(f"      Recall@{K_SEARCH}:     {avg_recall:.4f}")

    except Exception as e:
        print(f"EROARE in timpul procesarii modelului {model_name}: {e}")

print(f"\n--- Rezumat Comparativ ---")
print(f"Metrici calculate pentru top {K_SEARCH} rezultate returnate.")
print("-" * 80)
print(f"{'Model':<25} | {'Encoding (s)':<12} | {'Avg Search (s)':<14} | {'Hit Rate':<10} | {'Precision':<10} | {'Recall':<10}")
print("-" * 80)
for model_name, metrics in results.items():
    enc_time = f"{metrics['encoding_time']:.2f}" if metrics['encoding_time'] is not None else "N/A"
    search_time = f"{metrics['avg_search_time']:.4f}" if metrics['avg_search_time'] is not None else "N/A"

    if metrics['total_queries'] > 0 and metrics['avg_search_time'] is not None:
        hit_rate_val = metrics['hits_at_k'] / metrics['total_queries']
        precision_val = metrics['precision_at_k_sum'] / metrics['total_queries']
        recall_val = metrics['recall_at_k_sum'] / metrics['total_queries']
        hit_rate_str = f"{hit_rate_val:.2%}"
        precision_str = f"{precision_val:.4f}"
        recall_str = f"{recall_val:.4f}"
    else:
        hit_rate_str = "N/A"
        precision_str = "N/A"
        recall_str = "N/A"

    print(f"{model_name:<25} | {enc_time:<12} | {search_time:<14} | {hit_rate_str:<10} | {precision_str:<10} | {recall_str:<10}")
print("-" * 80)


# ============================================================
# STEP 3: ChromaDB Vector Store
# ============================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db_math"
COLLECTION_NAME = "math_problems"
SOURCE_COLUMN = "problem_cleaned"
METADATA_COLUMNS = ["solution_cleaned"]

print(f"\n{'='*60}")
print("STEP 3: ChromaDB Vector Store")
print(f"{'='*60}")

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"INFO: Se va folosi {'GPU (cuda)' if DEVICE == 'cuda' else 'CPU'}.")
except ImportError:
    DEVICE = "cpu"
    print("INFO: PyTorch nu este instalat, se va folosi CPU.")

print(f"\n1. Initializare model de embedding '{EMBEDDING_MODEL_NAME}'...")
start_time = time.time()
try:
    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': DEVICE},
        encode_kwargs={'normalize_embeddings': True}
    )
    print(f"   Model de embedding initializat pe {DEVICE}.")
except Exception as e:
    print(f"EROARE FATALA la initializarea modelului de embedding: {e}")
    exit()
embed_init_time = time.time() - start_time
print(f"   Timp initializare model: {embed_init_time:.2f} secunde.")

vectorstore = None
if os.path.exists(CHROMA_DB_PATH):
    print(f"\n2. Incarcare baza de date existenta din '{CHROMA_DB_PATH}'...")
    try:
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embedding_function,
            collection_name=COLLECTION_NAME
        )
        print(f"   ChromaDB incarcata cu succes. Contine {vectorstore._collection.count()} documente.")
    except Exception as e:
        print(f"   EROARE la incarcare: {e}. Se va crea o baza de date noua.")
        vectorstore = None
else:
    print(f"\n2. Directorul '{CHROMA_DB_PATH}' nu exista. Se va crea o baza de date noua.")

if vectorstore is None:
    print(f"\n2. Creare baza de date noua ChromaDB in '{CHROMA_DB_PATH}'...")
    start_time = time.time()
    try:
        loader = CSVLoader(
            file_path=output_csv_filename,
            source_column=SOURCE_COLUMN,
            metadata_columns=METADATA_COLUMNS,
            encoding='utf-8'
        )
        documents = loader.load()

        if not documents:
            raise ValueError("Nu s-au incarcat documente din CSV.")

        print(f"   Am incarcat {len(documents)} documente din CSV.")
        print("   Generare embeddings si indexare in ChromaDB (poate dura)...")

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_function,
            persist_directory=CHROMA_DB_PATH,
            collection_name=COLLECTION_NAME
        )
        print(f"   ChromaDB creata cu succes. Contine {vectorstore._collection.count()} documente.")

    except Exception as e:
        print(f"EROARE FATALA la crearea bazei ChromaDB: {e}")
        exit()
    create_db_time = time.time() - start_time
    print(f"   Timp creare baza de date: {create_db_time:.2f} secunde.")

print("\n3. Testare cautare semantica...")
if vectorstore:
    test_query = "solve linear equation 3x + 5 = 11"
    k_results = 3
    print(f"   Query de test: '{test_query}'")
    start_time = time.time()
    try:
        search_results = vectorstore.similarity_search(test_query, k=k_results)
        search_time = time.time() - start_time

        if search_results:
            print(f"   {len(search_results)} rezultate gasite in {search_time:.4f} secunde:")
            for i, doc in enumerate(search_results):
                print(f"      Rezultat {i+1}:")
                print(f"         Text: {doc.page_content[:150]}...")
                print(f"         Metadate: {doc.metadata}")
        else:
            print("   Nu s-au gasit rezultate similare.")
    except Exception as e:
        print(f"   EROARE la cautarea semantica: {e}")


# ============================================================
# STEP 4: RAG Pipeline (OpenAI GPT-3.5-turbo)
# ============================================================

print(f"\n{'='*60}")
print("STEP 4: RAG Pipeline")
print(f"{'='*60}")

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

llm = None
try:
    from google.colab import userdata
    openai_api_key = userdata.get('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError
    os.environ["OPENAI_API_KEY"] = openai_api_key
except ImportError:
    pass

OPENAI_MODEL_NAME = "gpt-3.5-turbo"

if "OPENAI_API_KEY" in os.environ:
    llm = ChatOpenAI(model_name=OPENAI_MODEL_NAME, temperature=0.1)
    print(f"   LLM initializat: {OPENAI_MODEL_NAME}")
else:
    print("EROARE: OPENAI_API_KEY nu este setat. Setati-l ca variabila de mediu.")
    exit()

prompt_template = """
You are an expert Python programmer specialized in mathematical libraries like NumPy and SymPy.
Use the provided context (examples of similar math problems and their Python solutions) to generate Python code that solves or represents the given mathematical problem described in natural language.

Context:
---------------------
{context}
---------------------

Mathematical Problem / Question:
{question}

Instructions:
1. Analyze the problem description.
2. Use relevant examples from the context if helpful.
3. Generate concise, runnable Python code using NumPy or SymPy.
4. Add minimal comments if necessary.
5. If you cannot solve it, state that clearly.

Python Code:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


def format_docs(docs):
    formatted_list = []
    for i, doc in enumerate(docs):
        content = doc.page_content
        solution = doc.metadata.get('solution_cleaned', 'N/A')
        formatted_list.append(
            f"Example {i+1}:\n"
            f"Problem Description: {content}\n"
            f"Solution:\n```python\n{solution}\n```"
        )
    return "\n\n".join(formatted_list) if formatted_list else "No relevant examples found in the context."


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

try:
    single_test_question = input(">>> Introduceti problema matematica (in limbaj natural) si apasati Enter: ")
    if not single_test_question or not single_test_question.strip():
        print("Nu a fost introdusa nicio intrebare.")
        exit()
    single_test_question = single_test_question.strip()
except (EOFError, KeyboardInterrupt):
    print("\nIntrerupt de utilizator.")
    exit()

start_testing_time = time.time()
generated_code = None
error_message = None

start_invoke_time = time.time()
try:
    generated_code = rag_chain.invoke(single_test_question)
    invoke_time = time.time() - start_invoke_time
except Exception as e:
    invoke_time = time.time() - start_invoke_time
    error_message = str(e)

print(f"\nIntrebare: {single_test_question}")
if error_message:
    print(f"EROARE: {error_message}")
elif generated_code:
    print(f"Cod Generat:\n```python\n{generated_code.strip()}\n```")
else:
    print("Nu a fost returnat niciun cod.")
print(f"Timp Generare: {invoke_time:.2f}s")
