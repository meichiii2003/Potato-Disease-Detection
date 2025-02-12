import streamlit as st
import tensorflow as tf
import numpy as np
import requests
from PIL import Image

# Load the trained Keras model
@st.cache_resource
def load_model():
    model_path = "saved_models/cnn_model.h5"
    return tf.keras.models.load_model(model_path)

cnn_model = load_model()

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
#geo_url = "http://ip-api.com/json/"


import streamlit as st

# Initialize session state for city and pop-up control
if "city" not in st.session_state:
    st.session_state["city"] = "-"  # Default city if not set
if "show_popup" not in st.session_state:
    st.session_state["show_popup"] = True  # Don't show pop-up on load

# Function to show the location pop-up
def show_location_popup():
    st.session_state["show_popup"] = True  # Trigger pop-up

# Sidebar: Display current location and button to open pop-up
with st.sidebar:
    if st.button("📍 Change Location"):
        show_location_popup()
    st.write(f"🌍 **Current Location:** {st.session_state['city']}")

    
# Show pop-up when triggered
if st.session_state["show_popup"]:
    with st.popover("🌍 Enter Your Location"):
        st.write("Please enter your city name for weather updates:")

        # Store input directly in session state
        city_name = st.text_input("City Name", value="", key="city_input").strip()

        # Update location when submitted
        if st.button("Submit"):
            if city_name:
                st.session_state["city"] = city_name  # ✅ Save new city
                st.session_state["show_popup"] = False  # Close pop-up
                st.rerun()  # Force update UI
            else:
                st.warning("⚠️ Please enter a valid city name.")

        # Remove location (reset to default)
        if st.button("Remove Location"):
            st.session_state["city"] = "-"
            st.session_state["show_popup"] = False  # Close pop-up
            st.rerun()  # Force UI update



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
pdf_text = extract_text_from_pdf("lec11.pdf")
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

# Use OpenWeather API to get latitude & longitude from the city name
if st.session_state["city"]:
    geocode_url = f"http://api.openweathermap.org/geo/1.0/direct?q={st.session_state["city"]}&limit=1&appid={api_key}"
    geocode_response = requests.get(geocode_url).json()
    
    if geocode_response:
        lat = geocode_response[0]["lat"]
        lon = geocode_response[0]["lon"]
    else:
        lat, lon = None, None
        st.error("⚠️ Could not retrieve location. Please enter a valid city name.")
else:
    lat, lon = None, None

# ✅ Ensure 3-Day Weather Summary exists
if "three_day_summary" not in st.session_state:
    st.session_state["three_day_summary"] = {}

# 🌦 Fetch Weather Forecast & Analyze Risks (RUNS AT STARTUP)
if st.session_state["city"]:
    temp_url = f"{forecast_url}appid={api_key}&lat={lat}&lon={lon}&units=metric"
    response = requests.get(temp_url)
    forecast_data = response.json()

    if forecast_data.get("cod") != "404":
        from datetime import datetime, timezone, timedelta
        forecast_list = forecast_data["list"]
        now = datetime.now(timezone.utc)  # Get current UTC time

        # Clear previous summary before updating
        st.session_state["three_day_summary"] = {}

        for forecast in forecast_list:
            dt_txt = forecast["dt_txt"]
            forecast_time = datetime.strptime(dt_txt, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

            # Convert to local time (Malaysia UTC +8)
            local_time = forecast_time + timedelta(hours=8)
            forecast_date = local_time.strftime('%Y-%m-%d')

            if local_time > now:  # Only future forecasts
                temp = forecast["main"]["temp"]  # Already in Celsius
                humidity = forecast["main"]["humidity"]
                wind_speed = forecast["wind"]["speed"]
                description = forecast["weather"][0]["description"]
                rain = forecast.get("rain", {}).get("3h", 0)  # Get rain in last 3 hours

                # 🌿 Risk Analysis
                alert_messages = []

                if temp < 15:
                    alert_messages.append("⚠️ **Cold Temperature Alert**: Risk of frost damage & slowed growth!")
                if temp > 32:
                    alert_messages.append("🔥 **Heat Stress Alert**: Protect plants from excessive sun exposure!")

                if humidity > 80:
                    alert_messages.append("🌫 **High Humidity Alert**: Increased risk of fungal diseases like blight!")

                if wind_speed < 1.0:
                    alert_messages.append("💨 **Low Wind Alert**: Stagnant air may promote fungal growth!")
                if wind_speed > 15:
                    alert_messages.append("🌪 **Strong Winds Alert**: Risk of plant damage, secure vulnerable crops!")

                if rain > 10:
                    alert_messages.append("🌧 **Heavy Rain Warning**: Possible waterlogging, ensure good drainage!")

                # 📝 Store alerts in session_state for ALL pages
                if forecast_date not in st.session_state["three_day_summary"]:
                    st.session_state["three_day_summary"][forecast_date] = set()  # Use set to avoid duplicates
                st.session_state["three_day_summary"][forecast_date].update(alert_messages)
    else:
        st.error("⚠️ Forecast data not found. Please check your internet connection or API key.")


st.sidebar.title("📍 Navigation")
page = st.sidebar.radio("Go to", [
    "🥔 Disease Detection", 
    "🤖 AI Assistant",
    "🌦 Current Weather", 
    "📅 Future Prediction"
])




# 📌 Get Location for Weather API
# geo_response = requests.get(geo_url).json()
# if geo_response.get("status") == "success":
#     city_name = geo_response["city"]
#     lat = geo_response["lat"]
#     lon = geo_response["lon"]
# else:
#     city_name = st.sidebar.text_input("Enter city name:")
#     lat, lon = None, None  # Ensure lat/lon exist

# 🌦 Current Weather Section
if page == "🌦 Current Weather":
    st.subheader("🌦 Weather Information")
    
    if lat and lon:
        temp_url = f"{base_url}appid={api_key}&lat={lat}&lon={lon}"
        response = requests.get(temp_url)
        weather_data = response.json()

        if weather_data.get("cod") != "404":
            main_data = weather_data["main"]
            current_temperature = main_data["temp"] - 273.15  # Convert to Celsius
            current_humidity = main_data["humidity"]
            wind_speed = weather_data["wind"]["speed"]
            weather_description = weather_data["weather"][0]["description"]

            st.write(f"**City:** {st.session_state["city"]}")
            st.write(f"🌡 **Temperature:** {current_temperature:.2f} C")
            st.write(f"💧 **Humidity:** {current_humidity}%")
            st.write(f"💨 **Wind Speed:** {wind_speed} m/s")
            st.write(f"🌥 **Condition:** {weather_description}")
        else:
            st.error("Weather data not found.")
    else:
        st.warning("⚠️ Unable to retrieve location. Please update your location at the side bar.")

# 📅 Future Weather Prediction Section
if page == "📅 Future Prediction":
    st.subheader("📅 Future Weather Forecast & Risk Analysis")

    # Define plant risk thresholds
    plant_risk_factors = {
        "high_humidity": 80,  # Risk of fungal diseases
        "low_temp": 15,       # Risk of slowed growth, frost
        "high_temp": 32,      # Risk of heat stress
        "low_wind": 1.0,      # Risk of fungal diseases (low air circulation)
        "heavy_rain": 10,     # High risk of waterlogging
        "storm_wind": 15      # Risk of plant damage
    }

    if lat and lon:
        temp_url = f"{forecast_url}appid={api_key}&lat={lat}&lon={lon}&units=metric"
        response = requests.get(temp_url)
        forecast_data = response.json()

        if forecast_data.get("cod") != "404":
            from datetime import datetime, timezone, timedelta
            forecast_list = forecast_data["list"]
            now = datetime.now(timezone.utc)  # Get current UTC time

            # Clear previous summary before updating
            st.session_state["three_day_summary"] = {}

            for forecast in forecast_list:
                dt_txt = forecast["dt_txt"]
                forecast_time = datetime.strptime(dt_txt, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

                # Convert to local time (Malaysia UTC +8)
                local_time = forecast_time + timedelta(hours=8)
                forecast_date = local_time.strftime('%Y-%m-%d')

                if local_time > now:  # Only future forecasts
                    temp = forecast["main"]["temp"]  # Already in Celsius
                    humidity = forecast["main"]["humidity"]
                    wind_speed = forecast["wind"]["speed"]
                    description = forecast["weather"][0]["description"]
                    rain = forecast.get("rain", {}).get("3h", 0)  # Get rain in last 3 hours

                    # 🌿 Risk Analysis
                    alert_messages = []

                    if temp < plant_risk_factors["low_temp"]:
                        alert_messages.append("⚠️ **Cold Temperature Alert**: Risk of frost damage & slowed growth!")
                    if temp > plant_risk_factors["high_temp"]:
                        alert_messages.append("🔥 **Heat Stress Alert**: Protect plants from excessive sun exposure!")

                    if humidity > plant_risk_factors["high_humidity"]:
                        alert_messages.append("🌫 **High Humidity Alert**: Increased risk of fungal diseases like blight!")

                    if wind_speed < plant_risk_factors["low_wind"]:
                        alert_messages.append("💨 **Low Wind Alert**: Stagnant air may promote fungal growth!")
                    if wind_speed > plant_risk_factors["storm_wind"]:
                        alert_messages.append("🌪 **Strong Winds Alert**: Risk of plant damage, secure vulnerable crops!")

                    if rain > plant_risk_factors["heavy_rain"]:
                        alert_messages.append("🌧 **Heavy Rain Warning**: Possible waterlogging, ensure good drainage!")

                    # 📝 Store alerts in session_state for ALL pages
                    if forecast_date not in st.session_state["three_day_summary"]:
                        st.session_state["three_day_summary"][forecast_date] = set()  # Use set to avoid duplicates
                    st.session_state["three_day_summary"][forecast_date].update(alert_messages)

                    # Display Forecast (Only inside Future Prediction Page)
                    st.markdown(f"### 📆 {local_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"🌡 **Temperature:** {temp:.2f}°C")
                    st.write(f"💧 **Humidity:** {humidity}%")
                    st.write(f"💨 **Wind Speed:** {wind_speed} m/s")
                    st.write(f"☁️ **Condition:** {description}")

                    # Display Alerts
                    if alert_messages:
                        for alert in alert_messages:
                            st.warning(alert)
                    else:
                        st.success("✅ No extreme weather risks detected!")

                    st.write("---")  # Divider for readability


        else:
            st.error("⚠️ Forecast data not found. Please check your internet connection or API key.")
    else:
        st.warning("⚠️ Unable to retrieve location. Please update your location in the sidebar.")






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
if page == "🥔 Disease Detection":
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
        prediction = cnn_model.predict(processed_image)
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

# ✅ Show Sidebar Summary on **ALL Pages**
# ✅ Global variable to store 3-day risk summary (Ensures it exists on all pages)
with st.sidebar.expander("⚠️ 3-Day Weather Risk Summary", expanded=False):
    if "three_day_summary" in st.session_state and st.session_state["three_day_summary"]:
        for date, alerts in st.session_state["three_day_summary"].items():
            st.write(f"📅 **{date}**")
            for alert in alerts:
                st.warning(alert)
    else:
        st.info("✅ No major weather risks in the next 3 days.")
