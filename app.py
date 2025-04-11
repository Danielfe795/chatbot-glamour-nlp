from flask import Flask, request, jsonify
from responder_modelo import generar_respuesta, extraer_palabras_clave_con_sinonimos, determinar_categoria, buscar_productos
import os

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    texto_usuario = data.get('mensaje', '')

    if not texto_usuario:
        return jsonify({"respuesta": "No se recibió mensaje."})

    keywords = extraer_palabras_clave_con_sinonimos(texto_usuario)
    categoria = determinar_categoria(texto_usuario)
    productos, mensaje = buscar_productos(keywords, categoria, buscar_similares=True)

    respuesta = generar_respuesta(texto_usuario)
    return jsonify({
        "respuesta": respuesta,
        "productos": productos,
        "mensaje_busqueda": mensaje
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
