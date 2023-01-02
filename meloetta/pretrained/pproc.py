import json
import torch

import pandas as pd
import plotly.express as px

from transformers import AutoTokenizer, AutoModel

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from tqdm import tqdm

descriptors = [
    "meloetta/js/data/BattlePokedex.json",
    "meloetta/js/data/BattleAbilities.json",
    "meloetta/js/data/BattleItems.json",
    "meloetta/js/data/BattleMovedex.json",
]


def main():

    model_id = "princeton-nlp/sup-simcse-roberta-large"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model = model.to(device)

    output_dim = 3
    pca = PCA(output_dim)
    kmeans = KMeans(25)
    scaler = StandardScaler()

    for file in descriptors:
        with open(file, "r") as f:
            data = json.load(f)

        texts = [json.dumps(a) for a in sorted(data.values(), key=lambda x: x["num"])]

        outputs = []
        with torch.no_grad():
            for text in tqdm(texts):
                encoded = tokenizer(
                    text, padding=True, truncation=True, return_tensors="pt"
                )
                for key in encoded:
                    if isinstance(encoded[key], torch.Tensor):
                        encoded[key] = encoded[key].to(device)
                outputs.append(model(**encoded).pooler_output.squeeze())

        vectors = torch.stack(outputs).cpu().numpy()
        labels = kmeans.fit_predict(vectors).astype(str)
        vectors = pca.fit_transform(vectors)
        vectors = scaler.fit_transform(vectors)

        df = [a for a in sorted(data.values(), key=lambda x: x["num"])]
        for i, s in enumerate(df):
            x, y, z = vectors[i]
            s["x"] = x
            s["y"] = y
            s["z"] = z
            s["cluster"] = labels[i]

        df = pd.DataFrame(df)
        fig = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="z",
            color="cluster",
            hover_data=["name"] + (["desc"] if "desc" in df.columns else []),
        )
        fig.show()

        name = file.split("/")[-1].split(".")[0]
        print(
            f"{name} - explained variance: {100 * sum(pca.explained_variance_ratio_):.2f}"
        )

        torch.save(vectors, f"pretrained/{name}.pt")


if __name__ == "__main__":
    main()
