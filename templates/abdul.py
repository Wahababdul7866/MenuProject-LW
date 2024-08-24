import random 
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer  # Import SimpleImputer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
from email.mime.text import MIMEText
import geocoder
import streamlit as st
import pyttsx3
import time
import smtplib
import googlesearch
import speech_recognition as sr
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import pywhatkit as kit
import os
from twilio.rest import Client
import psutil
from time import sleep
from pyautogui import locateOnScreen, click, press, typewrite
from webbrowser import open
from pathlib import Path
import cv2
import threading
import streamlit as st
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pywhatkit 
from twilio.rest import Client
from comtypes import CLSCTX_ALL, cast  # Import cast from comtypes
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import POINTER
import requests
from bs4 import BeautifulSoup
import requests
from bs4 import BeautifulSoup
import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.options import Options  
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import time
import numpy as np
import cv2
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def findLocation():
    g = geocoder.ip('me')
    latlng = g.latlng
    city = g.city
    state = g.state
    country = g.country
    st.write(f"Coordinates: {latlng}")
    st.write(f"City: {city}")
    st.write(f"State: {state}")
    st.write(f"Country: {country}")

    engine = pyttsx3.init()
    speech_text = f"You are currently located in {city}, {state}, {country}. The coordinates are {latlng}."
    engine.say(speech_text)
    engine.runAndWait()



def sendWhatsapp():
    """Sends a WhatsApp message using pywhatkit."""

    phone_number = st.text_input("Enter phone number (with country code):")
    message = st.text_area("Enter your message:")

    if st.button("Send WhatsApp"):
        try:
            # Send the message immediately
            now = time.localtime()
            pywhatkit.sendwhatmsg(phone_number, message, now.tm_hour, now.tm_min + 1)  # Send in 1 minute
            st.success("WhatsApp message sent successfully!")
        except Exception as e:
            st.error(f"Error sending WhatsApp message: {e}")

def sendBulkEmail():
    """
    Sends a bulk email using user-provided information.
    """
    try:
        # Get user inputs
        recipient_emails = st.text_area("Enter recipient email addresses (comma-separated):")
        subject = st.text_input("Enter subject:")
        body = st.text_area("Enter your message:")

        # Create the email message (with UTF-8 encoding)
        message = f'Subject: {subject}\n\n{body}'.encode('utf-8')  

        # Standardize line endings (after encoding)
        message = message.replace(b'\r\n', b'\n')  # Use b'' for bytes 

        # Connect to the SMTP server and send the email
        with smtplib.SMTP('smtp.gmail.com', 587) as ob:
            ob.ehlo()
            ob.starttls()
            ob.login('abdulwahabcyber1231@gmail.com', 'uxst uoap rwki gquz')
            ob.sendmail('abdulwahabcyber1231@gmail.com', recipient_emails.split(','), message)

        st.write("Email sent successfully!")

    except smtplib.SMTPAuthenticationError as e:
        st.write("Failed to authenticate:", e)
    except smtplib.SMTPRecipientsRefused as e:
        st.write("Failed to send email to recipient(s):", e)
    except Exception as e:
        st.write("Failed to send email:", e)



def sms_sender(phone, message):
    # Open the Google Messages web app
    open("https://messages.google.com/web/u/0/conversations/new")
    
    # Wait for the page to load
    sleep(10)

    # Locate the phone input box
    phone_box = locateOnScreen("phoneno.png", minSearchTime=60)
    if phone_box:
        click(phone_box)
        sleep(2)
            
        typewrite(phone)
        sleep(1)

        press('enter')
        sleep(1)

        # Locate the message input box
        message_box = locateOnScreen("message.png", minSearchTime=60)
        if message_box:
            click(message_box)
            sleep(1)

            typewrite(message)
            sleep(1)

            press('enter')
            sleep(1)
        else:
            st.write("Message input box not found.")
    else:
        st.write("Phone input box not found.")


def get_current_volume():
    """Gets the current system volume."""
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    return int(volume.GetMasterVolumeLevelScalar() * 100)  # Convert to percentage

def set_volume(volume_level):
    """Sets the system volume to the given level."""
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volume.SetMasterVolumeLevelScalar(volume_level / 100, None)  # Normalize to 0-1 range
    st.success("Volume set successfully!")

def text_to_speech():
    """Converts text to audio using pyttsx3."""
    text = st.text_area("Enter text to convert to speech:")
    if st.button("Speak"):
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        st.success("Text spoken successfully!")

def top5ResultGoogle(query):
    options = Options()
    options.add_argument('--headless')  # Run Chrome in headless mode
    options.add_argument('--disable-gpu')  # Disable GPU for headless mode
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

    driver = webdriver.Chrome(options=options)
    driver.get(f"https://www.google.com/search?q={query}") 
    driver.implicitly_wait(10)  

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    search_results = soup.find_all('div', class_='g')

    st.write("**Top 5 Search Results for: **" + query)
    st.write("-" * 30)

    if search_results:
        for result in search_results[:5]:
            title_element = result.find('h3', class_='LC20lb')
            if title_element:
                title = title_element.text
            else:
                title = "N/A"

            link_element = result.find('a', href=True)
            if link_element:
                link = link_element['href']
            else:
                link = "N/A"

            st.write(f"**Title:** {title}")
            st.write(f"**Link:** {link}")
            st.write("-" * 20) 
    else:
        st.write("No search results found.")
    
    driver.quit() 



def ownImgNumpy():
    car_image = np.zeros((100, 100, 3), dtype=np.uint8)
    background_color = [135, 206, 235]
    car_body_color = [255, 0, 0]
    window_color = [12, 24, 232]
    wheel_color = [0, 0, 0]
    car_image[:, :] = background_color
    car_image[60:80, 20:80] = car_body_color
    car_image[50:60, 30:70] = car_body_color
    car_image[55:60, 35:45] = window_color
    car_image[55:60, 55:65] = window_color
    car_image[80:85, 30:40] = wheel_color
    car_image[80:85, 60:70] = wheel_color
    
    # Convert NumPy array to PIL image
    pil_image = Image.fromarray(car_image)

    # Display the image using Streamlit
    st.image(pil_image)


def openNotepad():
    os.system("notepad")

def cutPhotoShowOnTop():
    def capture_image():
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            cv2.imwrite('captured_image.jpg', frame)
        return ret, frame

    def crop_image(image_path, left, top, right, bottom):
        image = Image.open(image_path)
        cropped_image = image.crop((left, top, right, bottom))
        cropped_image.save('cropped_image.png')
        return cropped_image

    def overlay_images(original_img_path, cropped_img_path, x_offset, y_offset):
        original_img = cv2.imread(original_img_path)
        cropped_img = Image.open(cropped_img_path).convert("RGBA")
        original_img_pil = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)).convert("RGBA")
        original_img_pil.paste(cropped_img, (x_offset, y_offset), cropped_img)
        combined = cv2.cvtColor(np.array(original_img_pil), cv2.COLOR_RGBA2BGRA)
        cv2.imwrite('overlayed_image.png', combined)
        return combined

    ret, frame = capture_image()
    if not ret:
        print("Failed to capture image")
        return

    height, width, _ = frame.shape
    left = width // 4
    top = height // 4
    right = left + 200
    bottom = top + 200

    crop_image('captured_image.jpg', left, top, right, bottom)
    x_offset = 50
    y_offset = 50
    result_image = overlay_images('captured_image.jpg', 'cropped_image.png', x_offset, y_offset)
    cv2.imshow('Overlayed Image', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def send_single_email():
    """Sends a single email using SMTP."""

    sender_email = st.text_input("Enter your email address:")
    sender_password = st.text_input("Enter your email password (use App Password if 2FA is enabled):", type="password")
    recipient_email = st.text_input("Enter recipient's email address:")
    subject = st.text_input("Enter subject:")
    message = st.text_area("Enter your message:")

    if st.button("Send Email"):
        try:
            # Create a secure connection with the server (using port 465)
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(sender_email, sender_password)
                
                # Create the email message
                msg = MIMEText(message)
                msg['Subject'] = subject
                msg['From'] = sender_email
                msg['To'] = recipient_email

                # Send the email
                server.sendmail(sender_email, recipient_email, msg.as_string())

                st.success("Email sent successfully!")
        except smtplib.SMTPAuthenticationError as e:
            st.error("Failed to authenticate:", e)
        except Exception as e:
            st.error(f"Failed to send email: {e}")



def apply_sunglasses(boy_image_path, sunglasses_image_path):
    """Applies sunglasses to a boy's image."""

    boy_image = cv2.imread(boy_image_path)
    sunglasses_image = cv2.imread(sunglasses_image_path, cv2.IMREAD_UNCHANGED)

    # Resize sunglasses to fit the boy's face (adjust as needed)
    scale_factor = 0.3  # Adjust this value to change the size of the sunglasses
    resized_sunglasses = cv2.resize(sunglasses_image, None, fx=scale_factor, fy=scale_factor)

    # Get the height and width of the resized sunglasses
    sunglasses_height, sunglasses_width, _ = resized_sunglasses.shape

    # Calculate the position to place the sunglasses (adjust as needed)
    x_offset = int(boy_image.shape[1] / 2 - sunglasses_width / 2) 
    # Adjust y_offset to place sunglasses on the eyes
    y_offset = int(boy_image.shape[0] / 2 - sunglasses_height / 1 - sunglasses_height / 4) 

    # Create a mask for the sunglasses (alpha channel)
    if sunglasses_image.shape[2] > 3:  # Check if there is an alpha channel
        alpha_mask = sunglasses_image[:, :, 3] / 255.0
        alpha_mask = np.expand_dims(alpha_mask, axis=2)
    else:
        alpha_mask = np.ones((sunglasses_image.shape[0], sunglasses_image.shape[1], 1), dtype=np.float32)

    # Resize the alpha_mask to match the shape of the resized_sunglasses
    alpha_mask = cv2.resize(alpha_mask, (sunglasses_width, sunglasses_height))

    # Apply the sunglasses to the boy's image (correct blending)
    for c in range(0, 3):
        boy_image[y_offset:y_offset + sunglasses_height, x_offset:x_offset + sunglasses_width, c] = (
            alpha_mask * resized_sunglasses[:, :, c] + (1 - alpha_mask) * boy_image[y_offset:y_offset + sunglasses_height, x_offset:x_offset + sunglasses_width, c]
        )

    # Convert the image to PIL format for display in Streamlit
    pil_image = Image.fromarray(cv2.cvtColor(boy_image, cv2.COLOR_BGR2RGB))

    return pil_image


def create_flower(size):
    """Generates a simple flower image."""
    flower_image = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = size // 3
    
    # Draw petals
    petal_color = [255, 192, 203]  # Light pink
    for i in range(5):
        angle = i * 2 * np.pi / 5
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] + radius * np.sin(angle))
        cv2.line(flower_image, center, (x, y), petal_color, 5)  # Adjust line thickness for petal size

    # Draw center circle
    center_color = [255, 0, 0]  # Red
    cv2.circle(flower_image, center, size // 8, center_color, -1) 
    
    pil_image = Image.fromarray(flower_image)
    st.image(pil_image)


def plot_data():
    """Creates a plot from user-provided data (either from text area or uploaded file)."""

    # Get data source from the user
    data_source = st.radio("Select data source:", ["Text Area", "Uploaded File"])

    # Get data from text area
    if data_source == "Text Area":
        data_input = st.text_area("Enter your data (comma-separated values, one row per line):", 
                                height=200)

        # Convert data into a list of lists
        data_rows = data_input.strip().splitlines()
        if not data_rows:
            st.warning("Please enter data in the text area.")
            return
        try:
            data = [row.split(",") for row in data_rows]
            data = [[float(value) for value in row] if len(row) > 1 else [float(row[0])] for row in data]
        except ValueError:
            st.error("Invalid data format. Please enter data as comma-separated values, one row per line.")
            return

    # Get data from uploaded file
    elif data_source == "Uploaded File":
        uploaded_file = st.file_uploader("Upload a CSV file:", type=["csv"])
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                data = data.values.tolist()  # Convert DataFrame to list of lists
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return
        else:
            st.warning("Please upload a CSV file.")
            return

    # Check if data has at least two rows
    if len(data) < 2:
        st.error("Please enter data with at least two rows.")
        return

    # Get plot type from the user
    plot_type = st.selectbox("Select plot type:", ["Line", "Scatter", "Bar", "Histogram"])

    # Create plot based on user selection
    if plot_type == "Line":
        if len(data[0]) == 2:
            x = [row[0] for row in data]
            y = [row[1] for row in data]
            plt.plot(x, y)
            st.pyplot(plt)
        else:
            st.error("Line plot requires data with two columns (x and y).")

    elif plot_type == "Scatter":
        if len(data[0]) == 2:
            x = [row[0] for row in data]
            y = [row[1] for row in data]
            plt.scatter(x, y)
            st.pyplot(plt)
        else:
            st.error("Scatter plot requires data with two columns (x and y).")

    elif plot_type == "Bar":
        if len(data[0]) == 2:
            x = [row[0] for row in data]
            y = [row[1] for row in data]
            plt.bar(x, y)
            st.pyplot(plt)
        else:
            st.error("Bar plot requires data with two columns (x and y).")

    elif plot_type == "Histogram":
        if len(data[0]) == 1:
            values = [row[0] for row in data]
            plt.hist(values)
            st.pyplot(plt)
        else:
            st.error("Histogram plot requires data with one column.")

    # Altair option for advanced plotting
    if st.checkbox("Use Altair for more interactive plots"):
        df = pd.DataFrame(data, columns=["x", "y"] if len(data[0]) == 2 else ["value"])
        if plot_type == "Line":
            chart = alt.Chart(df).mark_line().encode(x="x", y="y")
            st.altair_chart(chart)
        elif plot_type == "Scatter":
            chart = alt.Chart(df).mark_point().encode(x="x", y="y")
            st.altair_chart(chart)
        elif plot_type == "Bar":
            chart = alt.Chart(df).mark_bar().encode(x="x", y="y")
            st.altair_chart(chart)
        elif plot_type == "Histogram":
            chart = alt.Chart(df).mark_bar().encode(alt.X("value", bin=True), alt.Y("count()"))
            st.altair_chart(chart)
   
def pollution_prediction():

    df = pd.read_csv("pollution_data.csv")

    # Check and convert 'pp_feat' to numeric if needed
    if df['pp_feat'].dtype == object:
        df['pp_feat'] = pd.to_numeric(df['pp_feat'], errors='coerce')

    # Impute missing values in 'pp_feat'
    df['pp_feat'].fillna(df['pp_feat'].mean(), inplace=True)  

    # Select features and target variable
    features = st.multiselect("Select features for prediction:", df.columns)
    if features:
        X = df[features]
        y = df['pp_feat']  # Use 'pp_feat' as the target variable

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a Ridge regression model
        model = Ridge(alpha=st.number_input("Ridge Alpha (regularization strength):", value=0.1))
        model.fit(X_train, y_train)

        # Get user input for prediction
        user_input = {}
        for feature in features:
            user_input[feature] = st.number_input(f"Enter {feature}:", value=df[feature].mean())

        # Create a DataFrame from user input
        user_df = pd.DataFrame([user_input])

        # Make prediction
        prediction = model.predict(user_df)[0]

        # Display results
        st.write(f"Predicted pollution level: {prediction}")




def pollution_prediction1():
    st.title("Pollution Prediction App")

    # Load data
    df = pd.read_csv("pollution_data.csv")

    # Clean 'pp_feat' column 
    df['pp_feat'] = pd.to_numeric(df['pp_feat'], errors='coerce')  # Convert to numeric, handling non-numeric values
    df.dropna(subset=['pp_feat'], inplace=True)  # Remove rows with missing 'pp_feat'

    # Clean other features (optional)
    # Example: Handling outliers in 'temperature'
    # df = df[df['temperature'] < 100]  # Remove values above 100 degrees

    # Select features
    features = st.multiselect("Select features for prediction:", 
                           ['temperature', 'humidity', 'traffic_density', 'o3_mean', 'wind-speed_median',
                            'dew_median', 'no2_median', 'so2_median', 'pm10_median'])

    if features:
        X = df[features]
        y = df['pp_feat']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Impute missing values 
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # Choose Regularization
        reg_type = st.selectbox("Select regularization type:", ["Ridge", "Lasso"])
        if reg_type == "Ridge":
            model = Ridge(alpha=st.number_input("Ridge Alpha:", value=0.1))
        else:
            model = Lasso(alpha=st.number_input("Lasso Alpha:", value=0.1))

        # Train Model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Display results
        st.write("Mean Squared Error:", mse)
        st.write("R-squared:", r2)

        # Plot actual vs predicted values
        st.pyplot(plt.figure(figsize=(10, 6)))
        plt.scatter(y_test, y_pred)
        plt.xlabel("Actual Pollution Level")
        plt.ylabel("Predicted Pollution Level")
        plt.title("Actual vs Predicted Pollution Levels")


def get_weather(city):
    """Fetches weather data from OpenWeatherMap API."""
    api_key = "8d368aea9894b030e5b4c2b7aca2c515"  # Replace with your OpenWeatherMap API key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "appid=" + api_key + "&q=" + city
    response = requests.get(complete_url)
    if response.status_code == 200:
        data = response.json()
        temp_kelvin = data['main']['temp']
        temp_celsius = temp_kelvin - 273.15
        description = data['weather'][0]['description']
        st.write(f"Temperature: {temp_celsius:.2f}°C")
        st.write(f"Conditions: {description}")
    else:
        st.error("City not found or API error.")


def get_air_quality(city):
    """Fetches air quality data from PurpleAir API."""
    api_key = "8BB89239-33EA-11EF-95CB-42010A80000E"  # Replace with your PurpleAir API key
    base_url = "https://api.purpleair.com/v1/sensors"
    complete_url = f"{base_url}?api_key={api_key}&location={city}"
    response = requests.get(complete_url)
    if response.status_code == 200:
        data = response.json()
        if len(data['data']) > 0:
            pm25_value = data['data'][0]['pm2.5']
            st.write(f"PM2.5: {pm25_value}")
        else:
            st.write("No air quality data found for this location.")
    else:
        st.error("API Error.")


def get_ram_usage():
    """Fetches and displays information about RAM usage."""
    ram = psutil.virtual_memory()

    # Convert bytes to more readable units
    total_ram = humanize_bytes(ram.total)
    available_ram = humanize_bytes(ram.available)
    used_ram = humanize_bytes(ram.used)
    percent_used = ram.percent

    # Display the results in Streamlit
    st.write(f"Total RAM: {total_ram}")
    st.write(f"Available RAM: {available_ram}")
    st.write(f"Used RAM: {used_ram}")
    st.write(f"Percent Used: {percent_used}%")

def humanize_bytes(bytes):
    """Converts bytes to human-readable units (KB, MB, GB, etc.)."""
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(bytes) < 1024.0:
            return f"{bytes:.1f}{unit}B"
        bytes /= 1024.0
    return f"{bytes:.1f}YB"


def get_movie_recommendations(query):
    """Fetches movie recommendations from TMDb."""
    api_key = "0ccbd7354a881d9263ef3a3b432c5cbd"  # Get a free TMDb API key from https://developer.themoviedb.org/docs/getting-started
    base_url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": api_key, "query": query}
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        if data["results"]:
            recommendations = []
            for movie in data["results"][:5]:  # Get top 5 results
                recommendations.append(f"**{movie['title']}** ({movie['release_date']}) - {movie['overview']}")
            return "\n".join(recommendations)
        else:
            return "No movies found matching your query."
    else:
        return "Error fetching recommendations."


def play_rock_paper_scissors():
    """Plays a game of Rock Paper Scissors with the user."""
    st.title("Rock Paper Scissors!")

    # Get user's choice
    user_choice = st.selectbox("Choose your move:", ["Rock", "Paper", "Scissors"])

    # Computer's random choice
    computer_choice = random.choice(["Rock", "Paper", "Scissors"])

    # Display choices
    st.write(f"You chose: {user_choice}")
    st.write(f"Computer chose: {computer_choice}")

    # Determine the winner
    if user_choice == computer_choice:
        result = "It's a tie!"
    elif (
        (user_choice == "Rock" and computer_choice == "Scissors")
        or (user_choice == "Paper" and computer_choice == "Rock")
        or (user_choice == "Scissors" and computer_choice == "Paper")
    ):
        result = "You win!"
    else:
        result = "Computer wins!"

    st.write(f"Result: {result}")


def word_count(text):
    """Counts the number of words in a paragraph."""
    words = text.split()  # Split the text into words
    return len(words)

def celsius_to_fahrenheit(celsius):
    """Converts Celsius to Fahrenheit."""
    fahrenheit = (celsius * 9/5) + 32
    return fahrenheit

def create_dataframe():
    """Creates a DataFrame from user input."""
    st.title("Create DataFrame from User Input")

    # Get column names
    column_names = st.text_input("Enter column names (comma-separated):")
    if column_names:
        column_names = column_names.split(",")

        # Get data rows
        data_rows = st.text_area("Enter data rows (comma-separated values, one row per line):",
                                 height=200)
        if data_rows:
            data_rows = data_rows.strip().splitlines()
            data = []
            for row in data_rows:
                row_values = row.split(",")
                data.append(row_values)

            # Create the DataFrame
            df = pd.DataFrame(data, columns=column_names)

            # Display the DataFrame
            st.write(df)

            # Additional options:
            # - Save the DataFrame as a CSV file
            # - Perform operations on the DataFrame (e.g., analysis, plotting)
            # ...
        else:
            st.warning("Please enter data rows.")
    else:
        st.warning("Please enter column names.")


def linear_regression():
    """Performs linear regression on user-provided CSV data."""

    # Get CSV file from user
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return

        # Get x-axis and y-axis columns from user
        x_axis = st.selectbox("Select x-axis column (predictor):", data.columns)
        y_axis = st.selectbox("Select y-axis column (target):", data.columns)

        # Split data into training and testing sets
        X = data[[x_axis]]
        y = data[y_axis]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Display results
        st.subheader("Linear Regression Results")

        # Plot actual vs predicted values
        plt.scatter(X_test, y_test, label="Actual Values")
        plt.plot(X_test, y_pred, color="red", label="Predicted Values")
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.legend()
        st.pyplot(plt)

        # Display model coefficients
        st.write("**Coefficients:**")
        st.write(f"Intercept: {model.intercept_:.2f}")
        st.write(f"Slope: {model.coef_[0]:.2f}")

        # Display R-squared score
        st.write("**R-squared:**", round(model.score(X_test, y_test), 2))

        # User input for prediction (Fixed to accept a single value)
        user_input = st.number_input("Enter a value for " + x_axis + " to predict " + y_axis + ":")
        if user_input:
            # Predict using the user input (reshape to match the expected input shape)
            prediction = model.predict([[user_input]])  
            st.write(f"Predicted {y_axis}: {prediction[0]:.2f}")

    else:
        st.warning("Please upload a CSV file.")
 

def multiple_linear_regression():
    """Performs multiple linear regression on user-provided CSV data."""

    # Get CSV file from user
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return

        # Get predictor and target columns
        x_columns = st.multiselect("Select predictor columns:", data.columns)
        if not x_columns:
            st.warning("Please select at least one predictor column.")
            return
        y_axis = st.selectbox("Select target column:", data.columns)

        # Split data into training and testing sets
        X = data[x_columns]
        y = data[y_axis]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Impute missing values (using mean imputation here)
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)  # Apply the same imputation to the test set

        # Create and train the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Display results
        st.subheader("Multiple Linear Regression Results")

        # Plot actual vs predicted values (not ideal for multiple predictors, consider other visualizations)
        # ... (you can choose to plot based on one predictor if needed) 
        # ...

        # Display model coefficients
        st.write("**Coefficients:**")
        for i, coef in enumerate(model.coef_):
            st.write(f"{x_columns[i]}: {coef:.2f}")

        st.write(f"Intercept: {model.intercept_:.2f}")

        # Display R-squared score
        st.write("**R-squared:**", round(model.score(X_test, y_test), 2))

        # User input for prediction (multiple values)
        user_input_dict = {}
        for column in x_columns:
            user_input_dict[column] = st.number_input(f"Enter value for {column}:", key=column)

        if all(user_input_dict.values()):
            # Create a DataFrame from user input
            user_input_df = pd.DataFrame([user_input_dict])
            prediction = model.predict(user_input_df)
            st.write(f"Predicted {y_axis}: {prediction[0]:.2f}")

    else:
        st.warning("Please upload a CSV file.")


def logistic_regression():
    """Performs logistic regression on user-provided CSV data."""

    # Get CSV file from user
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return

        # Get predictor and target columns
        x_columns = st.multiselect("Select predictor columns:", data.columns)
        if not x_columns:
            st.warning("Please select at least one predictor column.")
            return
        y_axis = st.selectbox("Select target column (binary):", data.columns)

        # Split data into training and testing sets
        X = data[x_columns]
        y = data[y_axis]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Impute missing values (using mean imputation here)
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # Create and train the logistic regression model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Display results
        st.subheader("Logistic Regression Results")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        st.write("**Confusion Matrix:**")
        st.write(cm)

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"**Accuracy:** {accuracy:.2f}")

        # Display model coefficients
        st.write("**Coefficients:**")
        for i, coef in enumerate(model.coef_[0]):
            st.write(f"{x_columns[i]}: {coef:.2f}")

        st.write(f"Intercept: {model.intercept_[0]:.2f}")

        # User input for prediction (multiple values)
        user_input_dict = {}
        for column in x_columns:
            user_input_dict[column] = st.number_input(f"Enter value for {column}:", key=column)

        if all(user_input_dict.values()):
            # Create a DataFrame from user input
            user_input_df = pd.DataFrame([user_input_dict])
            prediction = model.predict(user_input_df)[0]  # Predict and get the first element
            st.write(f"Predicted {y_axis}: {prediction}")

    else:
        st.warning("Please upload a CSV file.")



def automatic_data_processing(file, target_column):
    """
    Automatically processes a dataset and trains a Linear Regression model.

    Parameters:
    - file: Uploaded file object from Streamlit
    - target_column: str, name of the target column for prediction

    Returns:
    - model: Trained model
    - metrics: Dictionary containing model evaluation metrics
    """
    try:
        data = pd.read_csv(file)
    except Exception as e:
        return f"Error loading dataset: {e}", None

    # Handle missing values
    data.fillna(method='ffill', inplace=True)  # Forward fill

    # Encode categorical variables if any
    categorical_columns = data.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Split data into features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    metrics = {
        'Mean Squared Error': mean_squared_error(y_test, y_pred),
        'R^2 Score': r2_score(y_test, y_pred)
    }

    return model, metrics


def apply_filter(frame, filter_type):
    if filter_type == "BLUR":
        return cv2.GaussianBlur(frame, (15, 15), 0)
    elif filter_type == "CONTOUR":
        return cv2.Canny(frame, 100, 200)
    elif filter_type == "DETAIL":
        # Use a sharpening filter as a proxy for DETAIL
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(frame, -1, kernel)
    elif filter_type == "EDGE_ENHANCE":
        # Use a simple edge enhancement filter
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        return cv2.filter2D(frame, -1, kernel)
    elif filter_type == "SHARPEN":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(frame, -1, kernel)
    else:
        return frame










