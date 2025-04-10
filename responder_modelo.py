from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import requests
import re
from urllib.parse import quote
from langdetect import detect

# Configuración inicial
modelo_path = r"C:\xampp\htdocs\GlamourSAS\modelo_entrenado"
tokenizer = AutoTokenizer.from_pretrained(modelo_path)
model = AutoModelForSeq2SeqLM.from_pretrained(modelo_path)

# Diccionario de preguntas frecuentes (simplificado para ejemplo)
preguntas_frecuentes = {
    "¿Tienen champús o acondicionadores para cabello seco?": "Sí, tenemos champús y acondicionadores específicamente diseñados para cabello seco. ¿Te gustaría ver algunos productos recomendados?",
    "¿Qué tipo de tratamientos capilares manejan?": "Ofrecemos una amplia gama de tratamientos capilares, como hidratación profunda, fortalecimiento, anticaída, revitalizantes, entre otros.",
    "¿Cuál es la diferencia entre un fluido y una crema para el cabello?": "Un fluido es más liviano y se absorbe rápidamente, mientras que una crema tiene una textura más espesa y es ideal para hidratación intensa.",
    "¿Tienen kits para cuidado del cabello?": "Sí, tenemos kits de cuidado capilar que incluyen productos combinados para tratamientos específicos.",
    "¿Venden lociones o sprays capilares?": "Sí, contamos con lociones y sprays diseñados para diferentes necesidades del cabello.",
    "¿Qué productos tienen para moldear rizos?": "Tenemos productos específicos para rizos, como geles, espumas y cremas para definir y mantener la forma de tus rizos.",
    "¿Qué me recomiendan para estimular el crecimiento de rizos?": "Para estimular el crecimiento de rizos, te recomendamos productos con ingredientes como keratina, aceites nutritivos y productos específicos para el cuidado del cabello rizado.",
    "¿Tienen productos anticaída para hombre?": "Sí, tenemos productos específicos para hombres que ayudan a prevenir la caída del cabello.",
    "¿Cuál es el mejor tratamiento revitalizante que ofrecen?": "Uno de los mejores tratamientos revitalizantes que ofrecemos es el tratamiento con colágeno, que rejuvenece el cabello y mejora su textura.",
    "¿Qué productos ayudan a controlar el frizz (Liss Control)?": "Contamos con productos como cremas y serums Liss Control para un control efectivo del frizz y suavidad en el cabello.",
    "¿Hay algún tratamiento sin enjuague que me recomienden?": "Sí, tenemos tratamientos sin enjuague como cremas y sprays que hidratan y protegen el cabello durante todo el día.",
    "¿Tienen productos con protección térmica para planchas?": "Sí, ofrecemos sprays y cremas con protección térmica para proteger tu cabello del calor de las planchas.",
    "¿Qué beneficios tiene la keratina vegana?": "La keratina vegana fortalece el cabello, lo hace más suave y brillante sin ingredientes animales, ideal para cabellos dañados o tratados químicamente.",
    "¿Tienen productos con colágeno o ceramidas?": "Sí, tenemos productos que incluyen colágeno y ceramidas, que son excelentes para reparar y fortalecer el cabello.",
    "¿Cuál es la diferencia entre la Keratin Ultra Force y la Vegan Keratin?": "La Keratin Ultra Force es una fórmula más potente para el cabello extremadamente dañado, mientras que la Vegan Keratin es más suave y natural, ideal para cabellos normales o ligeramente dañados.",
    "¿Qué productos tienen extracto de semilla de lino o durazno?": "Contamos con productos que contienen extracto de semilla de lino y durazno, ideales para nutrir y revitalizar el cabello.",
    "¿Tienen productos con ingredientes naturales como Green Forest?": "Sí, ofrecemos productos con extractos naturales como Green Forest, que proporcionan beneficios para la salud del cabello.",
    "¿Qué productos recomiendan para cabello normal?": "Para cabello normal, recomendamos champús y acondicionadores ligeros que mantengan el equilibrio de hidratación y suavidad.",
    "¿Tienen una línea especial para hombres?": "Sí, tenemos una línea exclusiva para hombres, que incluye champús, acondicionadores y tratamientos especializados.",
    "¿Qué productos son buenos para cabello teñido o con mechas radiantes?": "Contamos con productos diseñados para proteger y mantener el color del cabello teñido, como champús y tratamientos específicos para cabellos teñidos.",
    "¿Tienen algo para rizos definidos (Curls y Waves)?": "Sí, tenemos productos especializados para rizos definidos, como cremas y geles para definir y mantener la forma de los rizos.",
    "¿Qué recomiendan para proteger el color del cabello teñido (Color Guard)?": "Para proteger el color, recomendamos productos con tecnología Color Guard, que preservan el tono y la vitalidad del cabello teñido."
}

SINONIMOS = {
    "anticaspa": ["anticaspa", "caspa"],
    "antibacterial": ["antibacterial"],
    "brillo": ["brillante", "brillo", "reluciente"],
    "caída": ["antic caída", "caída", "cabello quebradizo", "fortalecimiento", "fortalecer", "revitalizante"],
    "cabello": [
        "cabello con mechas", "cabello crespo", "cabello graso", "cabello lacio", "cabello normal",
        "cabello ondulado", "cabello quebradizo", "cabello rizado", "cabello seco", "cabello teñido"
    ],
    "color": ["color guard", "mechas radiantes"],
    "crecimiento": ["alargar", "crecer", "crecimiento"],
    "estilo": ["estimulante de rizos", "frizz", "goma moldeadora", "liss control", "moldeadora", "spray", "termofijadora"],
    "formato": [
        "acondicionador", "ampolletas", "crema", "fluido", "gel", "infusión", "kit", "leave on", 
        "loción", "shampoo", "sin enjuague", "tratamiento"
    ],
    "hidratación": ["hidratación", "hidratar", "resequedad", "seco"],
    "ingredientes": [
        "ceramidas", "collagen", "durazno", "glicólica", "green forest", "keratin ultra force", 
        "lino", "semilla de lino", "ultractive", "vegan keratin"
    ],
    "puntas abiertas": ["maltratadas", "partidas", "puntas abiertas"],
    "reparación": ["recuperar", "reparación", "reparar", "restaurar"],
    "shampoo": ["champú", "limpiador", "shampoo"],
    "suavidad": ["sedoso", "suave", "suavidad"],
    "tratamiento": ["mascarilla", "terapia", "tratamiento"],
    "uso": ["hombre", "leave on", "long lasting", "protectoras", "sin enjuague"],
    "tipo de cabello": ["curls y waves", "liss control"]
}

CATEGORIAS = {
    "Shampoo": ["shampoo", "champú", "limpiador"],
    "Tratamiento": ["tratamiento", "mascarilla", "reparador", "terapia", "ampolla", "fluido", "loción"],
    "Acondicionador": ["acondicionador", "suavizante"],
    "Spray": ["spray", "atomizador", "bruma"],
    "Gel": ["gel", "fijador"],
    "Crema": ["crema", "leave-in", "sin enjuague"],
    "Kit": ["kit", "combo", "paquete"],
    "Loción": ["loción"],
}

STOPWORDS = set([
    "el", "la", "los", "las", "de", "para", "con", "un", "una",
    "me", "puedes", "quiero", "necesito", "ayuda", "por", "favor", "algun", "algún", "producto", "recomiéndame"
])

def detectar_idioma(texto):
    try:
        idioma = detect(texto)
        # print(f"Idioma detectado: {idioma}")  # solo si necesitás testear
        return idioma
    except:
        return "es"

def generar_respuesta(texto):
    """Genera una respuesta usando el modelo de lenguaje"""
    prompt = (
        "Eres un asistente experto en productos de belleza y cuidado personal. "
        "Responde de manera clara y breve. "
        "Usuario: " + texto + "\nAsistente:"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100, do_sample=True, top_p=0.95, top_k=60)
    
    respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    respuesta = respuesta.replace("<extra_id_0>", "").replace("<pad>", "").strip()

    return respuesta or "No estoy seguro de lo que estás preguntando. ¿Podrías reformular?"


def extraer_palabras_clave_con_sinonimos(texto):
    texto = texto.lower()
    palabras = re.findall(r'\b\w+\b', texto)
    palabras = [p for p in palabras if p not in STOPWORDS]

    keywords_detectadas = set()

    for palabra in palabras:
        for clave, sinonimos in SINONIMOS.items():
            if palabra in sinonimos:
                keywords_detectadas.add(clave)

    return list(keywords_detectadas)

def determinar_categoria(texto):
    texto = texto.lower()
    for categoria, sinonimos in CATEGORIAS.items():
        if any(re.search(rf'\b{re.escape(s)}\b', texto) for s in sinonimos):
            print("✅ Categoría detectada:", categoria)
            return categoria
    print("⚠️ No se encontró categoría para:", texto)
    return None


def buscar_productos(keywords, categoria, buscar_similares=False):
    """Consulta la API de productos, con opción de buscar productos similares si no se encuentra el solicitado."""
    try:
        if not keywords:
            return None, "No se proporcionaron palabras clave para la búsqueda."
        
        if not categoria:
            categoria = "Sin categoría"
        
        # Asegurarse de que keywords sea un string plano
        if isinstance(keywords, list):
            keywords = " ".join(keywords)
        keywords = str(keywords)

        # Confirmación visual
        print("➡️ Categoría final:", categoria)
        print("➡️ Keywords finales:", keywords)
        
        # Consulta el producto solicitado
        url = f"http://localhost/GlamourSAS/responder/buscar_producto.php?keyword={quote(keywords)}&categoria={quote(categoria)}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                productos = data.get('data', [])
                
                if productos:
                    return productos, ""
                
                # Si no se encuentra el producto exacto, buscar productos similares si se habilitó la opción
                if buscar_similares:
                    url_similares = f"http://localhost/GlamourSAS/responder/buscar_producto.php?keyword={quote(categoria)}"
                    response_similares = requests.get(url_similares, timeout=10)
                    if response_similares.status_code == 200:
                        data_similares = response_similares.json()
                        if data_similares.get('success'):
                            return data_similares.get('data', []), "No encontramos el producto exacto. Aquí tienes productos similares."
                
                return None, "No se encontraron productos con esos términos."
        
        return None, f"Error en la API (Código {response.status_code})"
    
    except Exception as e:
        return None, f"Error de conexión: {str(e)}"



def mostrar_productos(productos):
    if not productos:
        print("\n😕 No se encontraron productos para mostrar.")
        return

    print("\n🛍️ Productos encontrados:")
    for i, producto in enumerate(productos, start=1):
        nombre = producto.get("nombre", "Nombre desconocido")
        descripcion = producto.get("descripcion", "Sin descripción")
        precio = producto.get("precio", "Precio no disponible")
        stock = producto.get("stock", 0)
        categoria = producto.get("categoria", "Sin categoría")

        estado_stock = "✅ Disponible" if stock > 0 else "❌ Sin stock"

        print(f"\n🔸 Producto {i}:")
        print(f"📦 Nombre: {nombre}")
        print(f"🗂️ Categoría: {categoria}")
        print(f"💬 Descripción: {descripcion}")
        print(f"💰 Precio: ${precio}")
        print(f"📦 Stock: {stock} ({estado_stock})")

def responder_pregunta(entrada):
    """Responde preguntas frecuentes teniendo en cuenta sinónimos y variaciones"""
    entrada = entrada.lower().strip()
    for pregunta, respuesta in preguntas_frecuentes.items():
        # Compara la entrada con la pregunta y sus sinónimos
        if any(sinonimo in entrada for sinonimo in SINONIMOS.get(pregunta.lower(), [])) or pregunta.lower() in entrada:
            return respuesta
    return None

def validar_respuesta(respuesta):
    """Valida que la respuesta generada sea adecuada"""
    if not respuesta or len(set(respuesta.split())) < 3:
        return "No estoy seguro de lo que estás preguntando. ¿Podrías reformular?"
    return respuesta

def ejecutar_chatbot():
    print("\n" + "="*50)
    print("🤖 Asistente Virtual de Glamour SAS")
    print("="*50)
    print("\nPuedes preguntarme sobre nuestros productos de cuidado capilar.")
    print("Escribe 'salir' para terminar la conversación.\n")
    
    while True:
        try:
            entrada = input("Tú: ").strip()
            if not entrada:
                continue
                
            if entrada.lower() == 'salir':
                print("\n¡Gracias por consultar con nosotros! Hasta pronto.")
                break

            # 1. Verificar si es una pregunta frecuente
            respuesta = responder_pregunta(entrada)
            if respuesta:
                print(f"\nBot: {respuesta}")
                keywords = extraer_palabras_clave_con_sinonimos(entrada)
                categoria = determinar_categoria(entrada)
                productos, _ = buscar_productos(keywords, categoria)
                mostrar_productos(productos)
                continue

            # 2. Buscar productos primero
            keywords = extraer_palabras_clave_con_sinonimos(entrada)
            categoria = determinar_categoria(entrada)
            productos, mensaje = buscar_productos(keywords, categoria, buscar_similares=True)

            # 3. Construir respuesta según el resultado
            if productos:
                respuesta = generar_respuesta(entrada)
                print(f"\nBot: {respuesta}\n")
                mostrar_productos(productos)
            else:
                if any(palabra in entrada.lower() for palabra in [
                    "producto", "tratamiento", "sugerencia", "recomiéndame", "anticaspa", 
                    "shampoo", "crema", "caspa", "caída"
                ]):
                    respuesta = "No encontré productos con esas características, ¿quieres intentar con otra descripción?"
                else:
                    respuesta = generar_respuesta(entrada)

                print(f"\nBot: {respuesta}")
                print(f"\n🔍 {mensaje}")

        except KeyboardInterrupt:
            print("\n\n⚠️ Conversación interrumpida. Usa 'salir' para terminar adecuadamente.")
            continue
        except Exception as e:
            print(f"\n⚠️ Ocurrió un error inesperado: {str(e)}")
            continue

        print("\n¿En qué más puedo ayudarte?")


if __name__ == "__main__":
    ejecutar_chatbot()
