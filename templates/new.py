import abdul as ab
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_openml
import neuralnetwork as nn
import tensorflow as tf
import ml
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import ast 
import cv2
from PIL import Image 



st.set_page_config(layout="wide")  # Set layout to "wide" for larger buttons

st.markdown(
    """
    <div id="animation-container">
    
    </div>

    <script>
    const container = document.getElementById('animation-container');
    container.style.backgroundColor = 'lightblue';

    function animateBackground() {
        container.style.backgroundColor = 'lightpink';
        setTimeout(() => {
            container.style.backgroundColor = 'lightblue';
            setTimeout(animateBackground, 2000); // Animate every 2 seconds
        }, 2000);
    }
    animateBackground(); // Start the animation
    </script>
    """,
    unsafe_allow_html=True,
)
st.markdown(f"<h1 style='font-size: 48px; font-weight: bold; text-align: center; margin-bottom: 20px;'>----Menu--Project----</h1>", unsafe_allow_html=True)

option = st.selectbox(
    "WELCOME!!!!! What do you want to do",
    (
        "Current Location",
        "whatsapp",
        "Send Bulk Email",
        "SMS",
        "Volume",
        "Top 5 result",
        "texttoaudio",
        "ownimagenumpy",
        "Notepad",
        "Cutphotoshowontop",
        "Send Single Email",
        "Sunglasses Filter",
        "ownnumpyimage",
        "Word Count",
        "Create plot",
        "predict pollution",
        "predict pollution1",
        "Get Weather",
        "Celsius to Fahrenheit",
        "Get Air Quality",
        "Logistic Regression(Cancer)",
        "Get RAM Usage",
        "Movie Recommendations",
        "Game",
	  "Create DataFrame",
        "Linear regression",
        "Multiple linear regression",
        "Logistic regression",
	  "Automatic Data Processing",
        "Apply Live Video Filter",
 
    ),
    key="main_selectbox",  # Add a unique key for the selectbox
)

# --- Inline Styling for Buttons ---
def style_button(label, **kwargs):
    return st.button(
        f"<span style='font-size: 18px; padding: 15px 30px; border-radius: 5px; background-color: #4CAF50; color: white; border: none;'>{label}</span>",
        unsafe_allow_html=True,
        **kwargs,
    )


if option == "Current Location":
	ab.findLocation()
	
if option == "whatsapp":
	ab.sendWhatsapp()

if option == "Send Bulk Email":
    	ab.sendBulkEmail()

if option == "SMS":
    phone = st.text_input("Enter phone number:")  # Assign the input to 'phone'
    message = st.text_area("Enter message:")
    ab.sms_sender(phone, message)

if option == "texttoaudio":
    ab.text_to_speech()

if option == "Top 5 result":
    query = st.text_input("Enter your search query:") 
    ab.top5ResultGoogle(query)


if option == "Volume":
    # Create a container for volume control
    volume_container = st.container()

    # Get initial volume
    current_volume = ab.get_current_volume()

    # Create the slider within the container
    with volume_container:
        volume_level = st.slider("Set Volume", 0, 100, current_volume)

    # Set volume when the slider value changes
    volume_container.markdown("---")  # Add a separator line
    if volume_container.button("Set Volume"):
        ab.set_volume(volume_level)

if option == "ownimagenumpy":
	ab.ownImgNumpy()

if option == "Notepad":
	ab.openNotepad()

if option == "Cutphotoshowontop":
	ab.cutPhotoShowOnTop()

if option == "Send Single Email":  
      ab.send_single_email()

if option == "Sunglasses Filter":
    # Sunglasses Filter Logic
    st.title("Sunglasses Filter")

    # Upload images
    boy_image = st.file_uploader("Upload Boy Image", type=["jpg", "jpeg", "png"])
    sunglasses_image = st.file_uploader("Upload Sunglasses Image", type=["jpg", "jpeg", "png"])

    if boy_image and sunglasses_image:
        # Save uploaded images temporarily
        boy_image_path = "boy.png"
        sunglasses_image_path = "sunglasses.png"

        # Save boy's image
        with open(boy_image_path, "wb") as f:
            f.write(boy_image.read())

        # Save sunglasses image
        with open(sunglasses_image_path, "wb") as f:
            f.write(sunglasses_image.read())

        # Apply the filter and display the result
        result_image = ab.apply_sunglasses(boy_image_path, sunglasses_image_path)
        st.image(result_image)

if option == "ownnumpyimage":
    ab.create_flower(150)

if option == "Different imputation technique":
	st.title("Different Imputation Technique:")

	ml.show_long_text()

if option == "Weight info":
	st.title("Find what happens to the weight of dropped category in categorical variable:")

	ml.show_long_text1()

if option == "different initializers":
	st.title("about different initializers and their use cases")
	
	ml.show_long_text2()


if option == "optimizers":
	st.title("Find the use cases of optimizers")
	
	ml.show_long_text3()

if option == "activation":
	st.title("Find which activation function works with which type of pooling.")
	
	ml.show_long_text4()

if option == "Create plot":
	st.title("plot your data")

	ab.plot_data()

if option == "predict pollution":
	st.title("predict pollution")

	ab.pollution_prediction()

if option == "predict pollution1":
	st.title("predict pollution1")

	ab.pollution_prediction1()

if option == "Get Weather":
    city = st.text_input("Enter city name:")
    if st.button("Get Weather"):
        ab.get_weather(city)

if option == "Get Air Quality":
    city = st.text_input("Enter city name:")

    if st.button("Get Air Quality"):
        ab.get_air_quality(city)


if option == "Linear Regression (Boston Housing)":
    st.title("Linear Regression (Boston Housing)")
    
    # Load the Boston Housing dataset
    boston = fetch_openml(name="boston", as_frame=True)  
    X = boston.data
    y = boston.target
    
    # Select features and target variable
    features = st.multiselect("Select features for prediction:", X.columns)
    if features:
        X = X[features]
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Preprocessing (optional: scale features if needed)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)  

        # Get user input for prediction
        user_input = {}
        for feature in features:
            user_input[feature] = st.number_input(f"Enter {feature}:", value=X[feature].mean())

        # Create a DataFrame from user input
        user_df = pd.DataFrame([user_input])

        # Make prediction
        prediction = model.predict(scaler.transform(user_df[features]))[0]  

        # Display results
        st.write(f"Predicted value: {prediction}")

        # Plot actual vs predicted values (using st.pyplot correctly)
        plt.figure(figsize=(10, 6))  # Create the figure
        plt.scatter(y_test, y_pred)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted Values (Linear Regression)")
        plt.show()  # Show the plot within the Streamlit app
        st.pyplot()  # Display the plot in Streamlit

        

if option == "Logistic Regression(Cancer)":
    
    # Load the Breast Cancer dataset
    cancer = load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = cancer.target

    # Select features and target variable
    features = st.multiselect("Select features for prediction:", X.columns)
    if features:
        X = X[features]
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Preprocessing (optional: scale features if needed)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train the model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test) 

        # Get user input for prediction
        user_input = {}
        for feature in features:
            user_input[feature] = st.number_input(f"Enter {feature}:", value=X[feature].mean())

        # Create a DataFrame from user input
        user_df = pd.DataFrame([user_input])

        # Make prediction
        prediction = model.predict(scaler.transform(user_df[features]))[0]  

        # Display results
        st.write(f"Predicted class: {prediction}")

if option == "Get RAM Usage":
    ab.get_ram_usage()


if option == "Movie Recommendations":
    movie_query = st.text_input("Enter a movie title or keyword:")
    if st.button("Get Recommendations"):
        recommendations = ab.get_movie_recommendations(movie_query)
        st.write(recommendations)

if option =="Game":
	ab.play_rock_paper_scissors()

if option == "Share Output":
        st.title("Share Your App Output")
        content_type = st.selectbox("Select content type:", ["image", "chart"])

        if content_type == "image":
            uploaded_image = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"])
            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image")
                ab.share_on_social_media("image", image)
        elif content_type == "chart":
            # TODO: Implement chart sharing
            st.write("Chart sharing coming soon!")

if option == "Word Count":
    paragraph = st.text_area("Enter a paragraph:")
    if st.button("Count Words"):
        word_count_result = ab.word_count(paragraph)
        st.write(f"Word count: {word_count_result}")

if option == "Celsius to Fahrenheit":
    celsius_value = st.number_input("Enter temperature in Celsius:")
    if st.button("Convert"):
        fahrenheit_value = ab.celsius_to_fahrenheit(celsius_value)
        st.write(f"{celsius_value}°C is equal to {fahrenheit_value:.2f}°F")


if option == "Create DataFrame":
    ab.create_dataframe()

if __name__ == "__main__":
    
    if option == "Linear regression":
        ab.linear_regression() 
 

if __name__ == "__main__":

    if option == "Multiple linear regression":
        ab.multiple_linear_regression()


if __name__ == "__main__":
   
    if option == "Logistic regression":
        ab.logistic_regression()


if option == "Automatic Data Processing":
    st.title("Automatic Data Processing")
    
    file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    target_column = st.text_input("Enter the name of the target column:")

    if file and target_column:
        with st.spinner("Processing..."):
            model, metrics = automatic_data_processing(file, target_column)
            if metrics:
                st.write("Model Metrics:")
                st.write(metrics)
            else:
                st.write("Error processing the dataset.")



if option == "Apply Live Video Filter":
    st.title("Apply Live Video Filter")

    filter_type = st.selectbox(
        "Select a filter",
        ["None", "BLUR", "CONTOUR", "DETAIL", "EDGE_ENHANCE", "SHARPEN"],
        key="filter_selectbox"
    )

    if st.button("Start Video", key="start_video_button"):
        # Capture video from webcam
        cap = cv2.VideoCapture(0)

        # Streamlit components to display video
        stframe = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply selected filter
            if filter_type != "None":
                frame = ab.apply_filter(frame, filter_type)

            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)  # Convert frame to Image

            # Display video frame
            stframe.image(img, channels="RGB", use_column_width=True)

            # Stop video stream on button click
            if st.button("Stop Video", key="stop_video_button"):
                cap.release()
                break

        # Release video capture
        cap.release()
        cv2.destroyAllWindows()


