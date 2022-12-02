import torch
# prediction function
def predict(model, predict_dl, device):
  labels = []
  for x_batch, y_batch in predict_dl:
    inputs = x_batch.to(device)
    with torch.no_grad():
      output = model(inputs)
    labels += output[:, 0].tolist()
  return labels