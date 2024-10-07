import streamlit as st
import chromadb
from chromadb import Client
from chromadb.config import Settings
import google.generativeai as genai
from pypdf import PdfReader
import os, time

api_key = st.secrets["API_KEY"]  
genai.configure(api_key=api_key)
# Đường dẫn đến thư mục tạm
temp_dir = 'temp_uploaded_files'  # Tạo thư mục tạm ngay trong ứng dụng

# Thiết lập giao diện chính của trang
def page_setup():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #1d2671 0%, #c33764 100%);
        }
        .title-style {
            font-size: 40px;
            font-weight: bold;
            color: #ffffff;
            text-align: center;
            font-family: 'Helvetica Neue', sans-serif;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        .subtitle-style {
            font-size: 18px;
            color: #e1e1e1;
            text-align: center;
            font-family: 'Arial', sans-serif;
            margin-bottom: 30px;
        }
        .stButton button {
            background-color: #ff4b1f;
            color: white;
            border-radius: 20px;
            font-family: 'Arial', sans-serif;
            padding: 10px 20px;
            border: none;
            font-size: 16px;
            transition: background-color 0.3s, transform 0.2s;
        }
        .stButton button:hover {
            background-color: #ff9068;
            transform: scale(1.05);
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('<h1 class="title-style">Help Assistance Chat Bot 🤖</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="subtitle-style">Interact with PDFs, Images, Videos, and Audio files using Generative AI</h3>', unsafe_allow_html=True)


# Tạo danh sách để lưu lịch sử trò chuyện
chat_history = []
client = Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="chroma_db"))
collection = client.create_collection(name="chat_history")

def get_typeofpdf():
    st.sidebar.header("Select type of Media")
    typepdf = st.sidebar.radio("Choose one:",
                               ("📄 PDF files",
                                "🖼️ Images",
                                "🎥 Video, mp4 file",
                                "🎵 Audio files"))
    return typepdf

def get_llminfo():
    st.sidebar.header("Options", divider='rainbow')
    tip1="Select a model you want to use."
    model = st.sidebar.radio("Choose LLM:",
                                  ("gemini-1.5-flash",
                                   "gemini-1.5-pro",
                                   ), help=tip1)
    tip2="Lower temperatures are good for prompts that require a less open-ended or creative response, while higher temperatures can lead to more diverse or creative results."
    temp = st.sidebar.slider("Temperature:", min_value=0.0,
                                    max_value=2.0, value=1.0, step=0.25, help=tip2)
    tip3="Used for nucleus sampling. Specify a lower value for less random responses and a higher value for more random responses."
    topp = st.sidebar.slider("Top P:", min_value=0.0,
                             max_value=1.0, value=0.94, step=0.01, help=tip3)
    tip4="Number of response tokens, 8194 is limit."
    maxtokens = st.sidebar.slider("Maximum Tokens:", min_value=100,
                                  max_value=5000, value=2000, step=100, help=tip4)
    return model, temp, topp, maxtokens


def setup_temp_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def save_uploaded_file(uploaded_file, save_path):
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path

def add_to_chat_history(history, question, response):
    history.append({"question": question, "response": response})
    # Lưu vào ChromaDB
    collection.add(
        documents=[response],
        metadatas=[{"question": question}],
        ids=[str(len(history))]  # ID duy nhất cho mỗi bản ghi
    )


    
# Chức năng hiển thị lịch sử chat
def display_chat_history():
    st.sidebar.subheader("Chat History")
    results = collection.query(where={"question": {"$exists": True}}, n_results=5)  # Giới hạn số lượng kết quả
    for idx, doc in enumerate(results["documents"]):
        question = results["metadatas"][idx]["question"]
        if st.sidebar.button(f"Q{idx + 1}: {question}"):
            st.write(f"**Q: {question}**")
            st.write(f"**A: {doc}**")



def main():
    page_setup()
    typepdf = get_typeofpdf()
    model, temperature, top_p, max_tokens = get_llminfo()
    

    setup_temp_directory(temp_dir)
    
    if typepdf == "📄 PDF files":
        uploaded_files = st.file_uploader("Choose 1 or more PDF", type='pdf', accept_multiple_files=True)
        if uploaded_files:
            text = ""
            for pdf in uploaded_files:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text()

            generation_config = {
              "temperature": temperature,
              "top_p": top_p,
              "max_output_tokens": max_tokens,
              "response_mime_type": "text/plain",
              }
            model = genai.GenerativeModel(
              model_name=model,
              generation_config=generation_config)
            
            # Xử lý câu hỏi của người dùng
            question = st.text_input("Enter your question and hit return.")
            if question:
                response = model.generate_content([question, text])
                st.write(response.text)

                # Lưu câu hỏi và câu trả lời vào lịch sử
                add_to_chat_history(chat_history, question, response)
    
    elif typepdf == "🖼️ Images":
        image_file_name = st.file_uploader("Upload your image file.")
        if image_file_name:
            save_path = os.path.join(temp_dir, image_file_name.name)
            save_uploaded_file(image_file_name, save_path)

            # Upload file lên GenAI từ đường dẫn tạm thời
            image_file = genai.upload_file(path=save_path)

            # Kiểm tra trạng thái xử lý của file
            while image_file.state.name == "PROCESSING":
                time.sleep(10)
                image_file = genai.get_file(image_file.name)
            
            if image_file.state.name == "FAILED":
                raise ValueError(image_file.state.name)

            # Xử lý yêu cầu từ người dùng
            prompt = st.text_input("Enter your prompt.") 
            if prompt:
                generation_config = {
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_output_tokens": max_tokens,
                }
                model_instance = genai.GenerativeModel(model_name=model, generation_config=generation_config)
                response = model_instance.generate_content([image_file, prompt], request_options={"timeout": 600})
                st.markdown(response.text)
                add_to_chat_history(chat_history, prompt, response)

            # Xóa file sau khi xử lý
            genai.delete_file(image_file.name)
            os.remove(save_path)
    elif typepdf == "🎥 Video, mp4 file":
        video_file_name = st.file_uploader("Upload your video")
        if video_file_name:
            save_path = os.path.join(temp_dir, video_file_name.name)
            save_uploaded_file(video_file_name, save_path)
            video_file = genai.upload_file(path=save_path)
            
            while video_file.state.name == "PROCESSING":
                #st.write('.')
                time.sleep(10)
                video_file = genai.get_file(video_file.name)
            if video_file.state.name == "FAILED":
              raise ValueError(video_file.state.name)
            
            #file = genai.get_file(name=video_file.name)
            #st.write(f"Retrieved file '{file.display_name}' as: {video_file.uri}")
            
            # Create the prompt.
            prompt = st.text_input("Enter your prompt.") 
            if prompt:
                
                # The Gemini 1.5 models are versatile and work with multimodal prompts
                model = genai.GenerativeModel(model_name=model)
                
                # Make the LLM request.
                st.write("Making LLM inference request...")
                response = model.generate_content([video_file, prompt],
                                                  request_options={"timeout": 600})
                st.markdown(response.text)
                add_to_chat_history(chat_history, prompt, response)
                genai.delete_file(video_file.name)
                print(f'Deleted file {video_file.uri}')
                
    elif typepdf == "🎵 Audio files":
        audio_file_name = st.file_uploader("Upload your audio")
        if audio_file_name:
            save_path = os.path.join(temp_dir, audio_file_name.name)
            save_uploaded_file(audio_file_name, save_path)
            audio_file = genai.upload_file(path=save_path)

            while audio_file.state.name == "PROCESSING":
                time.sleep(10)
                audio_file = genai.get_file(audio_file.name)
            if audio_file.state.name == "FAILED":
              raise ValueError(audio_file.state.name)

            prompt = st.text_input("Enter your prompt.") #"what is said in this video in the first 20 seconds?"
            if prompt:
                model = genai.GenerativeModel(model_name=model)
                response = model.generate_content([audio_file, prompt],
                                                  request_options={"timeout": 600})
                st.markdown(response.text)
                add_to_chat_history(chat_history, prompt, response)
                genai.delete_file(audio_file.name)
                print(f'Deleted file {audio_file.uri}')
                
    display_chat_history(chat_history)


if __name__ == '__main__':
    main()
