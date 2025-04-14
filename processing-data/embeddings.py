import torch
import torchvision.transforms as T
from PIL import Image
import os
from tqdm import tqdm

# Папки
image_dir = "images"
embedding_dir = "embeddings"
os.makedirs(embedding_dir, exist_ok=True)

# Загрузка DINOv2 large
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)
model.eval()

# Преобразование изображений
transform = T.Compose([
    T.Resize(518),
    T.CenterCrop(518),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# Обработка и сохранение эмбеддингов
def process_image(img_path):
    filename = os.path.splitext(os.path.basename(img_path))[0]
    image = Image.open(img_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(img_tensor)  # [1, 1024]
    torch.save(embedding.cpu(), f"{embedding_dir}/{filename}.pt")

# Обработка всех изображений
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
for img_file in tqdm(image_files, desc="Processing embeddings"):
    try:
        process_image(os.path.join(image_dir, img_file))
    except Exception as e:
        print(f"Ошибка при обработке {img_file}: {e}")
