from flask import Flask, request, jsonify, current_app as application
import pickle
import numpy as np
from flask_cors import CORS, cross_origin
from os import path

app = Flask(__name__)

CORS(app,
     resources={r"/predict/*": { "origins": "https://trabalho-si-ai.onrender.com/" }},
     methods=['POST'],
     allow_headers=['Content-Type'],
     supports_credentials=True
)

@app.route('/predict', methods=['POST'])

@cross_origin()
def predict():
    filename = path.join(application.static_folder, 'data', 'predicao_de_casas_scaler.pkl')

    with open(filename, 'rb') as file:
        dados_salvos = pickle.load(file)
        modelo_treinado = dados_salvos['modelo']
        scaler = dados_salvos['scaler']

    data = request.get_json()
    if 'caracteristicas' not in data:
        return jsonify({'error': 'Os dados fornecidos são inválidos'}), 400

    caracteristicas_usuario = np.array(data['caracteristicas'])
    caracteristicas_normalizadas = scaler.transform([caracteristicas_usuario])

    valor_previsto = modelo_treinado.predict(caracteristicas_normalizadas)
    
    return jsonify({'valor_previsto': float(valor_previsto[0])})

if __name__ == '__main__':
    app.run(debug=True)