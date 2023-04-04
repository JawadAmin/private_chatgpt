from flask import Flask, request, render_template
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd
import numpy as np
from langchain import FewShotPromptTemplate
from langchain import PromptTemplate

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
    sorted_embeddings = clustered_text_df.sort_values("similarities", ascending=False).head(2)

    context = []
    for i, row in sorted_embeddings.iterrows():
        context.append(row['aggregated_text'][:1300])  # limit the number of tokens per matched sequence to 1300 tokens

    text = "\n".join(context)
    context = text
    
    # create our examples
    examples = [
        {
            "query": "How are you?",
            "answer": "I can't complain but sometimes I still do."
        }, {
            "query": "What is the law of demand?",
            "answer": "The law of demand is that the higher the price, the lower the quantity demanded, and the lower the price, the higher the quantity demanded. It's like the law of gravity, but for economics."
        }
    ]

    # create a example template
    example_template = """
    User: {query}
    AI: {answer}
    """

    # create a prompt example from above template
    example_prompt = PromptTemplate(
        input_variables=["query", "answer"],
        template=example_template
    )

    # now break our previous prompt into a prefix and suffix
    # the prefix is our instructions
    prefix = """The following are exerpts from conversations with an AI
    economics tutor. The tutor is typically sarcastic and witty, producing
    creative  and funny responses to the users questions. The tutor responds
    using the context provided. 

    Context: {context}

    Here are some response examples: 
    """
    # and the suffix our user input and output indicator
    suffix = """
    User: {query}
    AI: """

    # now create the few shot prompt template
    few_shot_prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["query", "context"],
        example_separator="\n\n"
    )

    user_prompt = f"""{few_shot_prompt_template.format(query=query, context=context)}"""
    system_prompt = f""""""

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
