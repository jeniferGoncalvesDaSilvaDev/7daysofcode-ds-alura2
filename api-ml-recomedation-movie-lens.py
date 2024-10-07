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
