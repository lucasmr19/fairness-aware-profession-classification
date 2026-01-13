import io
import torch
from torch.utils.data import Dataset as TorchDataset
import numpy as np
from PIL import Image

profession_classes = ["student", "blue_collar", "white_collar", "retired"]
profession_to_idx = {prof: i for i, prof in enumerate(profession_classes)}
idx_to_profession = {i: prof for prof, i in profession_to_idx.items()}
n_classes = len(profession_classes)

race_mapping = {
    0: "East Asian",
    1: "Indian",
    2: "Black",
    3: "White",
    4: "Middle Eastern",
    5: "Latino_Hispanic",
    6: "Southeast Asian",
}

gender_mapping = {
    0: "Male",
    1: "Female"
}


def bytes_to_pil(img_dict):
    """
    Converts a dict {'bytes': ..., 'path': ...} to PIL.Image
    """
    if isinstance(img_dict, dict) and 'bytes' in img_dict:
        return Image.open(io.BytesIO(img_dict['bytes'])).convert('RGB')
    else:
        return img_dict

# helper to unwrap lists/tuples/bytes safely
def _unwrap(value):
    # unwrap single-element lists/tuples
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    # decode bytes if needed
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode('utf-8')
        except Exception:
            pass
    return value

class FairFaceDataset(TorchDataset):
    def __init__(self, hf_dataset, transform=None, sensitive_attr='race'):
        """
        hf_dataset: huggingface datasets.Dataset
        transform: torchvision transforms
        sensitive_attr: name of the sensitive attribute (e.g., 'race')
        """
        self.ds = hf_dataset
        self.transform = transform
        self.sensitive_attr = sensitive_attr

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        example = self.ds[idx]

        # ---- IMAGE ----
        img_data = example.get('image', None)
        if img_data is None:
            raise KeyError(f"No 'image' field in example idx={idx}")

        img_data = _unwrap(img_data)

        if isinstance(img_data, Image.Image):
            img = img_data
        elif isinstance(img_data, dict) and 'bytes' in img_data:
            img = Image.open(io.BytesIO(img_data['bytes'])).convert('RGB')
        else:
            try:
                img = Image.open(io.BytesIO(img_data)).convert('RGB')
            except Exception as e:
                raise TypeError(f"Unsupported image format for idx={idx}: {type(img_data)}") from e

        if self.transform is not None:
            img = self.transform(img)

        # ---- LABEL ----
        label_raw = example.get('profession', None)
        if label_raw is None:
            raise KeyError(f"No 'profession' field in example idx={idx}")

        label_raw = _unwrap(label_raw)

        if isinstance(label_raw, (int, np.integer)):
            label_idx = int(label_raw)
        else:
            if not isinstance(label_raw, str):
                label_raw = str(label_raw)
            if label_raw not in profession_to_idx:
                raise ValueError(f"Unknown profession label '{label_raw}' at idx={idx}")
            label_idx = profession_to_idx[label_raw]

        # ---- SENSITIVE ATTRIBUTE ----
        sensitive_value = example.get(self.sensitive_attr, None)
        if sensitive_value is None:
            raise KeyError(f"No '{self.sensitive_attr}' field in example idx={idx}")
        if self.sensitive_attr == "race":
            sensitive_value = race_mapping[int(sensitive_value)]
        elif self.sensitive_attr == "gender":
            sensitive_value = gender_mapping[int(sensitive_value)]
            
        sensitive_value = _unwrap(sensitive_value)
        return img, torch.tensor(label_idx, dtype=torch.long), sensitive_value
