import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt")

class PizzaChatbot:
    def __init__(self):
        self.greetings = ["olá", "oi", "oi!", "olá!", "como vai?", "olá, como posso ajudar?"]
        self.goodbyes = ["adeus", "tchau", "até logo", "flw", "obrigado", "valeu"]
        self.menu = {
            "margherita": "A pizza Margherita é uma pizza simples com molho de tomate, queijo mussarela e folhas de manjericão.",
            "calabresa": "A pizza de calabresa tem molho de tomate, queijo mussarela e fatias de calabresa.",
            "portuguesa": "A pizza portuguesa tem molho de tomate, queijo mussarela, presunto, ovos, cebola, pimentão e azeitonas.",
            "pepperoni": "A pizza de pepperoni tem molho de tomate, queijo mussarela e fatias de pepperoni, que é um tipo de salame picante.",
            "vegetariana": "A pizza vegetariana tem molho de tomate, queijo mussarela e uma variedade de legumes, como tomate, cebola, pimentão e cogumelos."
        }
        self.vectorizer = TfidfVectorizer()
        self.vectorized_data = None

    def add_responses(self, responses):
        self.vectorized_data = self.vectorizer.fit_transform(responses)

    def get_response(self, input_text):
        input_vector = self.vectorizer.transform([input_text])
        similarities = cosine_similarity(input_vector, self.vectorized_data).flatten()
        best_match_index = np.argmax(similarities)
        
        if similarities[best_match_index] > 0.5:  # Aumentar o limite de similaridade para evitar respostas incorretas
            return list(self.menu.keys())[best_match_index], self.menu[list(self.menu.keys())[best_match_index]]
        else:
            return None, "Desculpe, não entendi. Poderia repetir?"

# Criar uma instância do chatbot
chatbot = PizzaChatbot()

# Alimentar o chatbot com respostas do menu
responses = list(chatbot.menu.values())
chatbot.add_responses(responses)

print("Olá! Bem-vindo à Pizzaria Delícia. Como posso ajudar você hoje?")
print("Dica: Experimente perguntar sobre nossas pizzas do menu!")

# Interagir com o chatbot
while True:
    input_text = input("Você: ").lower()
    
    if input_text in chatbot.greetings:
        print("Chatbot: Olá! Como posso ajudar?")
    elif input_text in chatbot.goodbyes:
        print("Chatbot: Até logo! Volte sempre.")
        break
    elif "menu" in input_text:
        menu_list = ", ".join(list(chatbot.menu.keys()))
        print(f"Chatbot: Temos as seguintes opções de pizza no nosso menu: {menu_list}.")
    else:
        pizza, resposta = chatbot.get_response(input_text)
        if pizza:
            print(f"Chatbot: A pizza {pizza.capitalize()} é deliciosa! {resposta}")
        else:
            print("Chatbot:", resposta)
