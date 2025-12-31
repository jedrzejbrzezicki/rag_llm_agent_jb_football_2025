from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

def create_rag_pipeline(db, model_name, tokenizer_name, prompt_template, max_new_tokens=256):
    """
    Creates a RAG pipeline using the given database, model, and prompt template.

    Args:
        db (Chroma): The Chroma database.
        model_name (str): Name of the model to use.
        tokenizer_name (str): Name of the tokenizer to use.
        prompt_template (str): The prompt template for the pipeline.
        max_new_tokens (int): Maximum number of tokens to generate.

    Returns:
        Callable: The RAG pipeline.
    """
    pipe = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer_name,
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True,
        return_full_text=False
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain