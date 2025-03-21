import numpy as np
import pickle

## loading the saved model
loaded_model = pickle.load(open("animalPrediction.csv", 'rb'))  


## creating a function for prediction
def animal_prediction(input_data):
    # دمج البيانات في نص واحد
    new_combined = ' '.join(input_data)
    
    # تحويل النص لـ TF-IDF vector باستخدام نفس الـ vectorizer
    new_transformed = vectorizer.transform([new_combined])
    
    # التنبؤ بالفئة
    prediction = loaded_model.predict(new_transformed)
    return prediction[0]


if prediction[0] == 'cat':
    print("The animal is a Cat")
elif prediction[0] == 'bird':
    print("The animal is a Bird")
elif prediction[0] == 'hamster':
    print("The animal is a Hamster")
elif prediction[0] == 'dog':
    print("The animal is a Dog")
else:
    print("Unknown animal")