import torch
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from constants import *

class PdfQA:
    def __init__(self, config):
        self.config = config

    def init_embeddings(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding = HuggingFaceEmbeddings(
            model_name=EMB_SBERT_MPNET_BASE,
            model_kwargs={"device": device}
        )

    def init_models(self):
        model = LLM_FLAN_T5_BASE
        tokenizer = AutoTokenizer.from_pretrained(model)
        llm_pipeline = pipeline(
            task="text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            model_kwargs={"device_map": "auto"}
        )
        self.llm = HuggingFacePipeline(pipeline=llm_pipeline)

    def vector_db_pdf(self):
        loader = PDFPlumberLoader(self.config["pdf_path"])
        docs = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = splitter.split_documents(docs)
        self.vectordb = Chroma.from_documents(texts, self.embedding)

    def retreival_qa_chain(self):
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="context: {context}\nquestion: {question}\nanswer:"
        )
        retriever = self.vectordb.as_retriever(search_kwargs={"k": 4})
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            chain_type="stuff"
        )
        self.qa.combine_documents_chain.llm_chain.prompt = prompt

    def answer_query(self, question):
        return self.qa({"query": question})["result"]
