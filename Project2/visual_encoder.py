import torch, pickle, tqdm
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from pathlib import Path

def save_features(model_name: str):
    search_dir = Path(Path.cwd(), "Assignment2/data/Animals_with_Attributes2/JPEGImages")
    image_dict = {}
    for image_class in search_dir.iterdir():
        for image_file in image_class.iterdir():
            image_dict[f"{image_class.stem}/{image_file.name}"] = str(image_file)

    data = {}
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to("cuda")
    with tqdm.tqdm(image_dict.items(), desc="Extracting features...", unit="images", colour="green") as pbar:
        for key, val in pbar:
            try:
                inputs = processor(images=Image.open(val), return_tensors="pt")
            except Exception as e:
                print(f"{e} occurred, skipping...")
                pass
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                last_hidden_states = outputs.pooler_output.squeeze().clone().detach()
            data[key] = {"feature": last_hidden_states, "label": key.split("/")[0]}

    pickle.dump(data, open(Path(Path.cwd(), f"Assignment2/data/visual_features/{model_name.split('/')[-1]}.pkl"), "wb"))

if __name__ == "__main__":
    model_name = "google/vit-base-patch16-224-in21k" # "microsoft/resnet-50"
    save_features(model_name=model_name)