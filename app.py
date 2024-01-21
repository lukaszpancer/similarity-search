import streamlit as st
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch
import numpy as np

st.markdown("# Similarity search")
st.markdown(
    "##### Enter a phrase to find relevant sentences. Feel free to input your own examples in the table below"
)


@st.cache_resource()
def get_model():
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
    return model, tokenizer


def get_embeddings(model, tokenizer, sentences: list[str]):
    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        model_output = model(**encoded_input)
        sentence_embeddings = model_output[0][:, 0]
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings


def calculate_similarity(
    embeddings: torch.Tensor, database_list: list, query: torch.Tensor
):
    similarity = np.inner(query, embeddings)
    ranked_database_list = [
        x for _, x in sorted(zip(similarity[0], database_list), reverse=True)
    ]
    return ranked_database_list[:3]


df = pd.DataFrame(
    {
        "sentences": [
            "Cole Fire rapidly grows to 236 acres in Barbados",
            "Leaked reports show shady plans to establish a Uruguayan ethnostate in Antarctica",
            "Scientists plan to kill owls to see why they die",
            "Man found dead in home, covered in 'white stuff'",
            "New virus found in Georgia, already killed 831",
            "Lior Barnes III under arrest for hair theft in Mexico",
            "Researchers plan to kill shrews to see why they die",
            "4.6 earthquake in Minnesota injures 1",
            "MacKenzie Scott becomes world's first trillionaire",
            "White supremacists kill 14 and injure 49 more in a North Carolina car incident",
        ]
    }
)

st.data_editor(
    df,
    num_rows="dynamic",
    use_container_width=True,
)

input_text = st.chat_input("Racist driver slaughters many in America")

if input_text:
    with st.spinner("Wait for it..."):
        model, tokenizer = get_model()
        embeddings = get_embeddings(model, tokenizer, df["sentences"].tolist())
        query = get_embeddings(model, tokenizer, [input_text])
        ranked_database_list = calculate_similarity(
            embeddings, df["sentences"].tolist(), query
        )
    st.markdown(
        f"""# Results:
#### 1. {ranked_database_list[0]}
#### 2. {ranked_database_list[1]}
#### 3. {ranked_database_list[2]}
"""
    )
