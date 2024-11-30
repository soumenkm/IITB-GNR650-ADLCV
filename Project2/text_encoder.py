import gensim.downloader
import pickle
from pathlib import Path

def save_features(model_name: str):
    search_dir = Path(Path.cwd(), "Assignment2/data/Animals_with_Attributes2/JPEGImages")
    image_dict = {}
    for image_class in search_dir.iterdir():
        for image_file in image_class.iterdir():
            image_dict[f"{image_class.stem}/{image_file.name}"] = str(image_file)

    classes = sorted(list(set([i.split("/")[0] for i in image_dict.keys()])))
    model = gensim.downloader.load(model_name)
    data = {}
    for c in classes:
        try:
            vector = model[c]
        except KeyError:
            vector = (model[c.split("+")[0]] + model[c.split("+")[1]])/2
        data[c] = vector

    pickle.dump(data, open(Path(Path.cwd(), f"Assignment2/data/text_features/{model_name.split('/')[-1]}.pkl"), "wb"))
    print("DONE")

if __name__ == "__main__":
    model_name = "fasttext-wiki-news-subwords-300" # "word2vec-google-news-300"
    save_features(model_name=model_name)