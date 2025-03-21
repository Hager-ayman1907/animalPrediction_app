import numpy as np
import pickle

## loading the saved model
loaded_model = pickle.load(open("C:/Users/Hagar/coursePython/Exercises with me & Youtube/animalPrediction.csv", 'rb'))  


x_new = ["pomeranian", "4", "small", "playful"]
new_combined = ' '.join(x_new)
new_transformed = vectorizer.transform([new_combined])

prediction = model.predict(new_transformed)
print("Prediction:", prediction[0])

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