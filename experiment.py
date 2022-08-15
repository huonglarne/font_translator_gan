from tqdm import tqdm

from src.constants import IMG_SIZE, STYLE_CHANNEL
from src.dataset.custom_dataset import FontDataset

train_dataset = FontDataset()
assert train_dataset is not None
assert len(train_dataset) == 753637

for i, data in enumerate(tqdm(train_dataset)):
    assert data["gt_images"].shape == (1, IMG_SIZE, IMG_SIZE)
    assert data["content_images"].shape == (1, IMG_SIZE, IMG_SIZE)
    assert data["style_images"].shape == (STYLE_CHANNEL, IMG_SIZE, IMG_SIZE)
    if i > 220:
        break


# gt_path
# './datasets/font/train/chinese/851CHIKARA-DZUYOKU-kanaB-2/一.png'

# english_font_path
# './datasets/font/train/english/851CHIKARA-DZUYOKU-kanaB-2'

# content_path
# './datasets/font/train/source/七.png'
