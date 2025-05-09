import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import streamlit as st
import time
from streamlit.components.v1 import html
from sklearn.metrics import accuracy_score
from PIL import Image

## loading the saved model and vectorizer
loaded_model = pickle.load(open("animalPrediction.csv", 'rb'))
vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))

## creating a function for prediction
def animal_prediction(input_data):
    # دمج البيانات في نص واحد
    new_combined = ' '.join(input_data)
    new_transformed = vectorizer.transform([new_combined])

    # التأكد من أن عدد الميزات متطابق
    if new_transformed.shape[1] != loaded_model.n_features_in_:
        raise ValueError(f"Expected {loaded_model.n_features_in_} features, but got {new_transformed.shape[1]}")

    # التنبؤ بالفئة
    prediction = loaded_model.predict(new_transformed)
    return prediction


# Function to render the initial cube animation with 6 cubes
def render_js_animation():
    animation_code = """
    <div class="container" style="position:relative; height:300px; display:flex; justify-content:center; align-items:center;">
        <div class="flex">
            <div class="cube">
                <div class="wall front"></div>
                <div class="wall back"></div>
                <div class="wall left"></div>
                <div class="wall right"></div>
                <div class="wall top"></div>
                <div class="wall bottom"></div>
            </div>
            <div class="cube">
                <div class="wall front"></div>
                <div class="wall back"></div>
                <div class="wall left"></div>
                <div class="wall right"></div>
                <div class="wall top"></div>
                <div class="wall bottom"></div>
            </div>
            <div class="cube">
                <div class="wall front"></div>
                <div class="wall back"></div>
                <div class="wall left"></div>
                <div class="wall right"></div>
                <div class="wall top"></div>
                <div class="wall bottom"></div>
            </div>
        </div>
        <div class="flex">
            <div class="cube">
                <div class="wall front"></div>
                <div class="wall back"></div>
                <div class="wall left"></div>
                <div class="wall right"></div>
                <div class="wall top"></div>
                <div class="wall bottom"></div>
            </div>
            <div class="cube">
                <div class="wall front"></div>
                <div class="wall back"></div>
                <div class="wall left"></div>
                <div class="wall right"></div>
                <div class="wall top"></div>
                <div class="wall bottom"></div>
            </div>
            <div class="cube">
                <div class="wall front"></div>
                <div class="wall back"></div>
                <div class="wall left"></div>
                <div class="wall right"></div>
                <div class="wall top"></div>
                <div class="wall bottom"></div>
            </div>
        </div>
    </div>
    <style>
        .flex {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 10px;
        }

        .cube {
            position: relative;
            width: 60px;
            height: 60px;
            transform-style: preserve-3d;
            animation: rotate 4s infinite;
            margin: 10px;
        }

        .wall {
            position: absolute;
            width: 60px;
            height: 60px;
            background: rgba(255, 255, 255, 0);
            border: 1px solid #000000 ;
        }

        .front { transform: translateZ(30px); }
        .back { transform: translateZ(-30px) rotateY(180deg); }
        .left { transform: rotateY(-90deg) translateX(-30px); transform-origin: center left; }
        .right { transform: rotateY(90deg) translateX(30px); transform-origin: center right; }
        .top { transform: rotateX(90deg) translateY(-30px); transform-origin: top center; }
        .bottom { transform: rotateX(-90deg) translateY(30px); }

        @keyframes rotate {
            0% { transform: rotateX(0deg) rotateY(0deg); }
            50% { transform: rotateX(180deg) rotateY(180deg); }
            100% { transform: rotateX(360deg) rotateY(360deg); }
        }
    </style>
    """
    st.markdown(animation_code, unsafe_allow_html=True)


# Function to render result animation
def render_result_animation(result):
    if result == 1:
        animation_code = """
        <div style="display:flex;justify-content:center;align-items:center;height:200px;">
            <div style="width:100px;height:100px;border-radius:50%;background-color:#ff5722;animation:bounce 1s infinite;"></div>
            <style>
                @keyframes bounce {
                    0%, 100% { transform: translateY(0); }
                    50% { transform: translateY(-20px); }
                }
            </style>
        </div>
        """
    else:
        animation_code = """
        <div style="display:flex;justify-content:center;align-items:center;height:200px;">
            <div style="width:100px;height:100px;border-radius:50%;background-color:#4caf50;animation:scale 1s infinite;"></div>
            <style>
                @keyframes scale {
                    0%, 100% { transform: scale(1); }
                    50% { transform: scale(1.2); }
                }
            </style>
        </div>
        """
    st.markdown(animation_code, unsafe_allow_html=True)


# عنوان التطبيق
st.title("🌟 Animal Prediction Web App! 🌟")

# عرض انيمشن المكعبات
st.write('"This tool uses machine learning models to predict the best animal for you based on your favorits..♥"')
render_js_animation()

def main():
    st.subheader("Enter your favorits:")

    # style change
    st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFB6B9;  
         color: #333333;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    # قوائم الاختيارات
    breed_name = ["campbell's dwarf", "chinese", "siamese", "pomeranian", "ragdoll", "winter white", "roborovski", "labrador", "cockatiel", "syrian", "maine coon", "lovebird", "chihuahua", "budgerigar", "parrot", "canary", "bulldog", "german shepherd", "persian", "bengal"]
    size = ["small", "large", "medium"]
    behavior = ['Curious', 'Playful', 'Shy', 'Calm', 'Lazy', 'Social', 'Noisy', 'Singing', 'Affectionate', 'Independent', 'Active', 'Energetic', 'Friendly', 'Loyal']

    # getting the input data from the user
    Breed = st.selectbox("Type of Breed of animal", breed_name)
    Age = st.selectbox("Age of animal", np.arange(1, 11, 1))
    Size = st.selectbox("Size of animal", size)
    Behavior = st.selectbox("Behavior of animal", behavior)

    # code for prediction
    adoption = ''
    img = ''

    # creating a button for prediction
    st.markdown(
            """
             <style>
            div.stButton > button:first-child {
            background-color: #5D4037;  
            color: #ffffff; 
            padding: 10px 20px;  
            font-size: 16px; 
            }
             </style>
             """,
         unsafe_allow_html=True
     )
    if st.button("Adoption Test Result"):
        Age = str(Age)
        
        st.write("Processing... Please wait.")
        with st.spinner("Predicting..."):
            time.sleep(2)

        prediction = animal_prediction([Breed, Age, Size, Behavior])
        predicted_animal = prediction[0] 
        animal_type = predicted_animal.split('(')[0]

        if animal_type.lower() == 'cat':
            adoption = f"The animal is a Cat with name {predicted_animal}"
            img = Image.open("cat.png")

        elif animal_type.lower() == 'bird':
            adoption =f"The animal is a Bird with name {predicted_animal}"
            img = Image.open("birds.png")
        elif animal_type.lower() == 'hamster':
            adoption =  f"The animal is a Hamester with name {predicted_animal}"
            img = Image.open("far.png")

        elif animal_type.lower() == 'dog':
            adoption =  f"The animal is a Dog with name {predicted_animal}"
            img = Image.open("dog.png")
        else:
            adoption = f"Unknown animal: {predicted_animal}"
        st.warning(adoption)
        # st.markdown(adoption, 
        #            """
        #             <div style="background-color: #F7BA57;">
        #             </div>
        #               """,   unsafe_allow_html=True
        #             )       
        st.image(img, caption='Predicted Animal')
        # عرض انيمشن النتيجة
        st.header("Prediction Result")
        render_result_animation(1)  # 1 لنتيجة معينة، 0 لنتيجة أخرى


if __name__ == '__main__':
    main()