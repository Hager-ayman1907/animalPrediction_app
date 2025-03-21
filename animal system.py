import numpy as np
import pickle

## loading the saved model
loaded_model = pickle.load(open("animalPrediction.csv", 'rb'))  


## creating a function for prediction
def animal_prediction(input_data):
    # دمج البيانات في نص واحد
   new_combined = ' '.join(x_new)
    
    # تحويل النص لـ TF-IDF vector باستخدام نفس الـ vectorizer
   new_transformed = vectorizer.transform([new_combined])
    
    # التنبؤ بالفئة
    prediction = loaded_model.predict(new_transformed)
    predicted_animal = prediction[0] 
    return predicted_animal


animal_type = predicted_animal.split('(')[0]

if animal_type.lower() == 'cat':
    print(f"The animal is a Cat with name {predicted_animal}")
elif animal_type.lower() == 'bird':
    print(f"The animal is a Bird with name {predicted_animal}")
elif animal_type.lower() == 'hamster':
    print(f"The animal is a Hamster with name {predicted_animal}")
elif animal_type.lower() == 'dog':
    print(f"The animal is a Dog with name {predicted_animal}")
else:
    print(f"Unknown animal: {predicted_animal}")