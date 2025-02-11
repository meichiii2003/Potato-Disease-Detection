import streamlit as st
import tensorflow as tf
import numpy as np
import requests
from PIL import Image

# Load the trained Keras model
@st.cache_resource
def load_model():
    model_path = "C://Users//meich//OneDrive - Asia Pacific University//Machine Learning and Data Science Project//Potato Disease Classification//saved_models//1.keras"
    return tf.keras.models.load_model(model_path)

model = load_model()

# Define class labels based on training dataset
CLASS_NAMES = ["Healthy", "Early Blight", "Late Blight"]

# Disease suggestions
DISEASE_SUGGESTIONS = {
    "Healthy": "✅ Your plant is healthy! Keep monitoring and maintain proper watering and sunlight.",
    "Early Blight": "⚠️ Early Blight detected! Remove affected leaves, apply copper-based fungicides, and avoid overhead watering.",
    "Late Blight": "🚨 Late Blight detected! Immediately remove and destroy infected plants. Apply fungicides like chlorothalonil and improve air circulation."
}

# Weather API settings
api_key = "4788fe1bcb4e6ea22f28f3ce7334a943"
base_url = "http://api.openweathermap.org/data/2.5/weather?"
forecast_url = "http://api.openweathermap.org/data/2.5/forecast?"
geo_url = "http://ip-api.com/json/"





import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

# 📌 Function to Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# Example Usage - Extract Text
pdf_text = extract_text_from_pdf("C://Users//meich//Downloads//lec11.pdf")
#print(pdf_text[:1000])  # Print first 1000 characters to verify extraction

# 📌 Function to Chunk Text
def chunk_text(text, chunk_size=512):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Create Chunks from Extracted Text
chunks = chunk_text(pdf_text, 512)
print(f"Total Chunks: {len(chunks)}")

# 📌 Load Embedding Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert Text Chunks into Embeddings
embeddings = np.array([model.encode(chunk) for chunk in chunks]).astype('float32')

# 📌 Create and Save FAISS Index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "potato_diseases.index")
print("FAISS index created and saved.")

# 📌 Function to Retrieve Relevant Text Chunk
def retrieve_relevant_chunk(query, index, chunks, model):
    query_embedding = np.array([model.encode(query)]).astype('float32')
    _, top_k = index.search(query_embedding, 1)  # Retrieve top match
    return chunks[top_k[0][0]]  # Return the most relevant chunk

# Example Query for Retrieval
query = "What are the symptoms of late blight in potatoes?"
relevant_chunk = retrieve_relevant_chunk(query, index, chunks, model)
# print(f"Retrieved Info:\n{relevant_chunk}")

# 📌 AI Assistant - TatoGuardAI with Gemini API
client = genai.Client(api_key="AIzaSyDoja20TTkp2v2472wWBqJOjZ2Ag8sZYUg")

instruction = (
        "You are TatoGuardAI, a chatbot designed to provide accurate and informative responses related to potato diseases. It ensures that all answers are clear, structured, and easy to understand, focusing only on potato health, disease management, and prevention. Responses should be direct and natural, without referencing any specific source. When explaining diseases, symptoms, causes, and management strategies, concise paragraphs should be used. However, for steps, best practices, or treatment options, the chatbot should present them in point format for clarity. When answering a question about a specific potato disease, TatoGuardAI should follow a structured approach by stating the disease name and its causative organism, describing the symptoms affecting leaves, stems, tubers, or roots, and highlighting early warning signs for early detection. It should then explain the mode of spread and survival, including whether transmission occurs through infected tubers, contaminated soil, air, or insect vectors, while also mentioning environmental factors such as temperature, humidity, and soil conditions that favor disease progression. Management and control strategies should include preventive measures such as using disease-free seed tubers, practicing crop rotation, applying fungicides, and implementing soil management techniques. If resistant potato varieties exist, they should be recommended, along with best agricultural practices such as proper irrigation, storage precautions, and pest control. For general potato health and disease prevention, TatoGuardAI should provide practical insights such as ideal planting conditions, methods for improving soil health, and strategies for early disease detection. If a user asks for a comparison between two diseases, the chatbot should clearly outline the differences in symptoms, spread, and management in a logical and structured manner. When multiple treatment options exist, it should provide a balanced explanation of their effectiveness, limitations, and recommendations. If the query is about a specific pathogen, such as Phytophthora infestans or Alternaria solani, the chatbot should describe its characteristics, how it affects potatoes, and the best control methods. TatoGuardAI should always remain factual and precise, avoiding unnecessary elaboration or speculation. If a question is unclear, it should ask for clarification before responding, and if a question falls outside its scope, it should politely indicate that the requested information is unavailable. The language should remain professional yet accessible, avoiding overly technical jargon unless the user specifically requests scientific details. If a user asks an irrelevant, nonsensical, or completely off-topic question, TatoGuardAI should respond politely and professionally without engaging in unrelated discussions. It should acknowledge the query, clarify that it does not fall under its expertise, and gently guide the user back to potato-related topics. Possible responses include polite redirection, such as stating that the chatbot is focused on potato diseases and offering to assist with relevant queries, clarification requests to ensure the user's question is related to potato health, or professional declines when the topic is outside the chatbot’s scope. In cases of lighthearted or humorous queries, TatoGuardAI may provide a friendly but professional response, maintaining engagement while keeping the conversation relevant. For confidential topics, TatoGuardAI should avoid answering and respond in a polite but neutral manner, either by changing the subject or expressing uncertainty without engaging in discussions on sensitive matters. If a user insists on off-topic or disruptive questions, the chatbot should remain courteous and restate its purpose without further engagement, always ensuring a respectful, professional, and encouraging tone for a helpful and focused user experience. By following these instructions, TatoGuardAI will provide reliable, well-structured, and professional responses while effectively handling off-topic, irrelevant, or confidential inquiries in a polite manner."
    )

# Example Query to AI
response = client.models.generate_content(
    model="gemini-1.5-flash", 
    contents=["Provide an overview of late blight disease in potatoes, including symptoms, causes, and treatment."],  # Ensure correct format
    config=types.GenerateContentConfig(
        temperature=1.5
    )
)

# Display AI Response
# print(response.text)






# Streamlit Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Current Weather", "Future Prediction", "Disease Detection", "🤖 AI Assistant"])

# 📌 Get Location for Weather API
geo_response = requests.get(geo_url).json()
if geo_response.get("status") == "success":
    city_name = geo_response["city"]
    lat = geo_response["lat"]
    lon = geo_response["lon"]
else:
    city_name = st.sidebar.text_input("Enter city name:")
    lat, lon = None, None  # Ensure lat/lon exist

# 🌦 Current Weather Section
if page == "Current Weather":
    st.subheader("🌦 Weather Information")
    
    if lat and lon:
        temp_url = f"{base_url}appid={api_key}&lat={lat}&lon={lon}"
        response = requests.get(temp_url)
        weather_data = response.json()

        if weather_data.get("cod") != "404":
            main_data = weather_data["main"]
            current_temperature = main_data["temp"]
            current_humidity = main_data["humidity"]
            wind_speed = weather_data["wind"]["speed"]
            weather_description = weather_data["weather"][0]["description"]

            st.write(f"**City:** {city_name}")
            st.write(f"🌡 **Temperature:** {current_temperature:.2f} K")
            st.write(f"💧 **Humidity:** {current_humidity}%")
            st.write(f"💨 **Wind Speed:** {wind_speed} m/s")
            st.write(f"🌥 **Condition:** {weather_description}")
        else:
            st.error("Weather data not found.")
    else:
        st.warning("Unable to retrieve location. Please enter your city manually.")

# 📅 Future Weather Prediction Section
if page == "Future Prediction":
    st.subheader("📅 Future Weather Forecast")

    if lat and lon:
        temp_url = f"{forecast_url}appid={api_key}&lat={lat}&lon={lon}"
        response = requests.get(temp_url)
        forecast_data = response.json()

        if forecast_data.get("cod") != "404":
            forecast_list = forecast_data["list"][:5]  # First 5 timestamps
            for forecast in forecast_list:
                dt_txt = forecast["dt_txt"]
                temp = forecast["main"]["temp"]
                description = forecast["weather"][0]["description"]
                st.write(f"📆 **{dt_txt}**")
                st.write(f"🌡 Temperature: {temp:.2f} K")
                st.write(f"🌥 Condition: {description}")
                st.write("---")
        else:
            st.error("Forecast data not found.")
    else:
        st.warning("Unable to retrieve location. Please enter your city manually.")

# 📅 AI Assistant Section
if page == "🤖 AI Assistant":
    st.title("🤖 TatoGuardAI - Potato Disease Chatbot")
    st.write("Ask any question about potato diseases, symptoms, prevention, and treatment!")
    
    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
     
    client = genai.Client(api_key="AIzaSyDoja20TTkp2v2472wWBqJOjZ2Ag8sZYUg")

    instruction = (
        "You are TatoGuardAI, a chatbot designed to provide accurate and informative responses related to potato diseases. It ensures that all answers are clear, structured, and easy to understand, focusing only on potato health, disease management, and prevention. Responses should be direct and natural, without referencing any specific source. When explaining diseases, symptoms, causes, and management strategies, concise paragraphs should be used. However, for steps, best practices, or treatment options, the chatbot should present them in point format for clarity. When answering a question about a specific potato disease, TatoGuardAI should follow a structured approach by stating the disease name and its causative organism, describing the symptoms affecting leaves, stems, tubers, or roots, and highlighting early warning signs for early detection. It should then explain the mode of spread and survival, including whether transmission occurs through infected tubers, contaminated soil, air, or insect vectors, while also mentioning environmental factors such as temperature, humidity, and soil conditions that favor disease progression. Management and control strategies should include preventive measures such as using disease-free seed tubers, practicing crop rotation, applying fungicides, and implementing soil management techniques. If resistant potato varieties exist, they should be recommended, along with best agricultural practices such as proper irrigation, storage precautions, and pest control. For general potato health and disease prevention, TatoGuardAI should provide practical insights such as ideal planting conditions, methods for improving soil health, and strategies for early disease detection. If a user asks for a comparison between two diseases, the chatbot should clearly outline the differences in symptoms, spread, and management in a logical and structured manner. When multiple treatment options exist, it should provide a balanced explanation of their effectiveness, limitations, and recommendations. If the query is about a specific pathogen, such as Phytophthora infestans or Alternaria solani, the chatbot should describe its characteristics, how it affects potatoes, and the best control methods. TatoGuardAI should always remain factual and precise, avoiding unnecessary elaboration or speculation. If a question is unclear, it should ask for clarification before responding, and if a question falls outside its scope, it should politely indicate that the requested information is unavailable. The language should remain professional yet accessible, avoiding overly technical jargon unless the user specifically requests scientific details. If a user asks an irrelevant, nonsensical, or completely off-topic question, TatoGuardAI should respond politely and professionally without engaging in unrelated discussions. It should acknowledge the query, clarify that it does not fall under its expertise, and gently guide the user back to potato-related topics. Possible responses include polite redirection, such as stating that the chatbot is focused on potato diseases and offering to assist with relevant queries, clarification requests to ensure the user's question is related to potato health, or professional declines when the topic is outside the chatbot’s scope. In cases of lighthearted or humorous queries, TatoGuardAI may provide a friendly but professional response, maintaining engagement while keeping the conversation relevant. For confidential topics, TatoGuardAI should avoid answering and respond in a polite but neutral manner, either by changing the subject or expressing uncertainty without engaging in discussions on sensitive matters. If a user insists on off-topic or disruptive questions, the chatbot should remain courteous and restate its purpose without further engagement, always ensuring a respectful, professional, and encouraging tone for a helpful and focused user experience. By following these instructions, TatoGuardAI will provide reliable, well-structured, and professional responses while effectively handling off-topic, irrelevant, or confidential inquiries in a polite manner."
    )

    # Display previous conversations
    st.subheader("💬 Chat History")
    for chat in st.session_state.chat_history:
        st.write(f"**You:** {chat['user_query']}")
        st.write(f"**TatoGuardAI:** {chat['response']}")
        st.write("---")
    
    # User input form to prevent immediate rerun issues
    with st.form("chat_form", clear_on_submit=True):
        user_query = st.text_input("Ask your question here:", key="user_query")
        submit_button = st.form_submit_button("Send")

    if submit_button and user_query:
        # 🔥 Append the instruction to the query
        full_prompt = instruction + "\n\nUser Query: " + user_query

        response = client.models.generate_content(
            model="gemini-1.5-flash", contents=full_prompt, 
            config=types.GenerateContentConfig(
            temperature=1.7
            )
        )
        
        # Store the conversation in session state
        st.session_state.chat_history.append({
            "user_query": user_query,
            "response": response.text
        })
        
        # Remove user query from session state before rerun
        st.session_state.pop("user_query", None)


        # Refresh the page to update the chat history
        st.rerun()
        
        #st.write("**You:**", user_query)
        #st.write("**TatoGuardAI:**", response.text)
        
# 🥔 Potato Disease Detection Section
if page == "Disease Detection":
    st.title("🥔 Potato Leaf Disease Detection")
    st.write("Upload an image of a potato leaf to detect disease!")

    uploaded_file = st.file_uploader("📂 Upload a leaf image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        # Load and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        def preprocess_image(img):
            img = img.resize((256, 256))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            return img_array

        processed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(processed_image)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]

        # Display Prediction Results
        st.subheader("🔍 Prediction Result")
        st.write(f"**Detected:** {predicted_class}")
        st.subheader("💡 Suggestion")
        st.write(DISEASE_SUGGESTIONS[predicted_class])

        # Display confidence scores
        st.write("📊 Confidence Scores:")
        for i, class_name in enumerate(CLASS_NAMES):
            st.write(f"{class_name}: {prediction[0][i]*100:.2f}%")


st.info("Ensure the uploaded image is clear and properly formatted.")
