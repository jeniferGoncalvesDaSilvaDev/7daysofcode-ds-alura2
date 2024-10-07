# instala a biblioteca
!pip install surprise
#no jupyter notebook usa-se a exclamação a frente no pip 
# Importar as bibliotecas necessárias
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise.model_selection import cross_validate

# Exemplo de um dataset de filmes com avaliações (MovieLens)
# Aqui usamos um exemplo do Surprise, mas você pode carregar o seu próprio dataset
# O dataset deve estar no formato: user_id, item_id, rating
data = Dataset.load_builtin('ml-100k')  # Exemplo com MovieLens 100k

# Dividir os dados em conjunto de treino e teste
trainset, testset = train_test_split(data, test_size=0.25)

# Escolher o algoritmo. Vamos usar SVD (Singular Value Decomposition)
model = SVD()

# Treinar o modelo
model.fit(trainset)

# Fazer previsões no conjunto de teste
predictions = model.test(testset)

# Avaliar o modelo com RMSE (Root Mean Squared Error)
accuracy.rmse(predictions)

# Função para fazer recomendações para um usuário específico
def get_top_n_recommendations(predictions, user_id, n=5):
    # Filtrar as previsões para o usuário solicitado
    user_predictions = [pred for pred in predictions if pred.uid == user_id]

    # Ordenar por maior nota prevista
    user_predictions.sort(key=lambda x: x.est, reverse=True)

    # Retornar os top N itens recomendados
    top_n = user_predictions[:n]
    for pred in top_n:
        print(f"Item {pred.iid} - Estimativa de Avaliação: {pred.est:.2f}")

# Fazer recomendações para um usuário (Ex: user_id = 196)
get_top_n_recommendations(predictions, user_id='196', n=5)

# Salvar o modelo treinado (adicione este bloco de código)
import pickle 

with open('svd_recommender_model.pkl', 'wb') as f:
    pickle.dump(model, f)


#!pip install surprise
#!pip install fastapi
from fastapi import FastAPI
import pickle
from surprise import Dataset, Reader

# Inicializar a aplicação FastAPI
app = FastAPI()

# Carregar o modelo treinado
with open('svd_recommender_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Função para gerar previsões para um usuário específico
def get_top_n_recommendations(user_id, n=5):
    # Carregar o dataset para pegar os ids dos itens (caso precise)
    data = Dataset.load_builtin('ml-100k')
    trainset = data.build_full_trainset()

    # Fazer previsões para todos os itens para o usuário específico
    user_inner_id = trainset.to_inner_uid(user_id)
    item_ids = trainset.all_items()

    predictions = []
    for item_id in item_ids:
        raw_id = trainset.to_raw_iid(item_id)
        prediction = model.predict(user_id, raw_id)
        predictions.append(prediction)

    # Ordenar as previsões por maior estimativa
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Retornar os top N itens recomendados
    top_n = predictions[:n]
    return [{'item_id': pred.iid, 'rating_est': pred.est} for pred in top_n]

# Endpoint da API para obter recomendações
@app.get("/recommendations/{user_id}")
def recommendations(user_id: str, n: int = 5):
    try:
        top_n_recommendations = get_top_n_recommendations(user_id, n)
        return {"user_id": user_id, "recommendations": top_n_recommendations}
    except Exception as e:
        return {"error": str(e)}
