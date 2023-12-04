import os
from inference import model_fn, input_fn, predict_fn, output_fn

os.environ['CURL_CA_BUNDLE'] = '' # for ssl error

with open('sample.txt', mode='rb') as file:
    model_input_data = file.read()
model = model_fn()
transformed_inputs = input_fn(model_input_data)
predicted_classes_jsonlines = predict_fn(transformed_inputs, model)
model_outputs = output_fn(predicted_classes_jsonlines)
print(model_outputs[0])
