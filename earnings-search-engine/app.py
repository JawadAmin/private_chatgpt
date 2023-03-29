from flask import Flask, request, render_template
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd
import numpy as np

app = Flask(__name__)
app.static_folder = 'static'

COMPLETIONS_MODEL = "gpt-3.5-turbo"
EMBEDDINGS_MODEL = "text-embedding-ada-002"
openai.api_key = ""

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

def chatbot_response(msg):
    res = getResponse(msg)
    return res

def getResponse(query):
    question_vector = get_embedding(query, engine=EMBEDDINGS_MODEL)

    clustered_text_df = pd.read_csv('clustered_embeddings.csv')
    clustered_text_df['embedding'] = clustered_text_df['embedding'].apply(eval).apply(np.array)
    clustered_text_df["similarities"] = clustered_text_df['embedding'].apply(lambda x: cosine_similarity(x, question_vector))
    sorted_embeddings = clustered_text_df.sort_values("similarities", ascending=False).head(3)

    context = []
    for i, row in sorted_embeddings.iterrows():
        context.append(row['aggregated_text'][:1300])  # limit the number of tokens per matched sequence to 1300 tokens

    text = "\n".join(context)
    context = text
    
    system_prompt = f"""Answer the following question using only the context provided. Answer in the style of a financial analyst. If you don't know the answer for certain, say I don't know."""
    user_prompt = f"""
    Context:
    {context}

     Q: {query}
    A:"""

    response = openai.ChatCompletion.create(
        temperature=0.5,
        max_tokens=700,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        model=COMPLETIONS_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )["choices"][0]["message"]
    
    result = response.content
    return result

if __name__ == '__main__':
    app.run(debug=True)
