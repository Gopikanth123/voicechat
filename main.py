import os  
import shutil  
from flask import Flask, render_template, request, jsonify  
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, SimpleDirectoryReader, ChatPromptTemplate, Settings  
from llama_index.llms.huggingface import HuggingFaceInferenceAPI  
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  
from huggingface_hub import InferenceClient  

# Ensure HF_TOKEN is set  
HF_TOKEN = os.getenv("HF_TOKEN")  
if not HF_TOKEN:  
    raise ValueError("HF_TOKEN environment variable not set.")  

repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"  
llm_client = InferenceClient(  
    model=repo_id,  
    token=HF_TOKEN,  
)  

# Configure Llama index settings  
Settings.llm = HuggingFaceInferenceAPI(  
    model_name=repo_id,  
    tokenizer_name=repo_id,  
    context_window=3000,  
    token=HF_TOKEN,  
    max_new_tokens=512,  
    generate_kwargs={"temperature": 0.1},  
)  
Settings.embed_model = HuggingFaceEmbedding(  
    model_name="BAAI/bge-small-en-v1.5"  
)  

PERSIST_DIR = "db"  
PDF_DIRECTORY = 'data'  

# Ensure directories exist  
os.makedirs(PDF_DIRECTORY, exist_ok=True)  
os.makedirs(PERSIST_DIR, exist_ok=True)  
chat_history = []  
current_chat_history = []  

def data_ingestion_from_directory():  
    # Clear previous data by removing the persist directory  
    if os.path.exists(PERSIST_DIR):  
        shutil.rmtree(PERSIST_DIR)  # Remove the persist directory and all its contents  
    
    # Recreate the persist directory after removal  
    os.makedirs(PERSIST_DIR, exist_ok=True)  
    
    # Load new documents from the directory  
    new_documents = SimpleDirectoryReader(PDF_DIRECTORY).load_data()  
    
    # Create a new index with the new documents  
    index = VectorStoreIndex.from_documents(new_documents)  
    
    # Persist the new index  
    index.storage_context.persist(persist_dir=PERSIST_DIR)  

def handle_query(query):  
    chat_text_qa_msgs = [  
        (  
            "user",  
            """  
            You are the Taj Hotel chatbot, known as Taj Hotel Helper. Your goal is to provide accurate and professional answers to user queries based on the information available about the Taj Hotel. Always respond clearly and concisely, ideally within 10-15 words. If you don't know the answer, say so politely.  
            Context:  
            {context_str}  
            User's Question:  
            {query_str}  
            """  
        )  
    ]  
    
    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)  
    
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)  
    index = load_index_from_storage(storage_context)  
    context_str = ""  
    
    # Build context from current chat history  
    for past_query, response in reversed(current_chat_history):  
        if past_query.strip():  
            context_str += f"User asked: '{past_query}'\nBot answered: '{response}'\n"  

    query_engine = index.as_query_engine(text_qa_template=text_qa_template, context_str=context_str)  
    print(f"Querying: {query}")  
    answer = query_engine.query(query)  

    # Extracting the response  
    if hasattr(answer, 'response'):  
        response = answer.response  
    elif isinstance(answer, dict) and 'response' in answer:  
        response = answer['response']  
    else:  
        response = "I'm sorry, I couldn't find an answer to that."  

    # Append to chat history  
    current_chat_history.append((query, response))  
    return response

app = Flask(__name__)  

# Data ingestion  
data_ingestion_from_directory()  

# Generate Response  
def generate_response(query):  
    try:  
        # Call the handle_query function to get the response  
        bot_response = handle_query(query)  
        return bot_response  
    except Exception as e:  
        return f"Error fetching the response: {str(e)}"  

# Route for the homepage  
@app.route('/')  
def index():  
    return render_template('index.html')  

# Route to handle chatbot messages  
@app.route('/chat', methods=['POST'])  
def chat():  
    try:  
        user_message = request.json.get("message")  
        if not user_message:  
            return jsonify({"response": "Please say something!"})  

        bot_response = generate_response(user_message)  
        return jsonify({"response": bot_response})  
    except Exception as e:  
        return jsonify({"response": f"An error occurred: {str(e)}"})  

if __name__ == '__main__':  
    app.run(debug=True)