import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device)

optimizer = optim.AdamW(sam.mask_decoder.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()  # Можно использовать BCEWithLogitsLoss
num_epochs = 10

for epoch in range(num_epochs):
    sam.train()
    total_loss = 0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        # Получение эмбеддингов изображения
        with torch.no_grad():
            image_embeddings = sam.image_encoder(images)

        # Генерация масок
        mask_pred = sam.mask_decoder(image_embeddings)

        # Вычисление потерь
        loss = criterion(mask_pred, masks)
        total_loss += loss.item()

        # Обновление весов
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")
