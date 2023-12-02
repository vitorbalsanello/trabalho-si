from flask import Flask, request, jsonify
import sys
import pickle
import numpy as np
sys.path.append('c:/users/admin/appdata/local/packages/pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0/localcache/local-packages/python311/site-packages')
from flask_cors import CORS
from flask_cors import cross_origin

app = Flask(__name__)
CORS(app, resources={r"/predict/*": {"origins": "http://localhost:8080"}},
         methods=['POST'], 
         allow_headers=['Content-Type'],
         supports_credentials=True)

with open('C:/Users/admin/Desktop/predicao_de_casas_scaler.pkl', 'rb') as file:
    dados_salvos = pickle.load(file)
    modelo_treinado = dados_salvos['modelo']
    scaler = dados_salvos['scaler']

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    data = request.get_json()
    if 'caracteristicas' not in data:
        return jsonify({'error': 'Os dados fornecidos são inválidos'}), 400

    caracteristicas_usuario = np.array(data['caracteristicas'])
    caracteristicas_normalizadas = scaler.transform([caracteristicas_usuario])

    valor_previsto = modelo_treinado.predict(caracteristicas_normalizadas)
    
    return jsonify({'valor_previsto': float(valor_previsto[0])})

if __name__ == '__main__':
    app.run(debug=True)