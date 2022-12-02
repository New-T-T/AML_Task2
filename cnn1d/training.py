import torch
import time
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
PATH = '/content/drive/MyDrive/task2'

def prediction_metrics(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    y_test_array = y_test.numpy(force=True)
    y_pred_tags_array = y_pred_tags.numpy(force=True)

    accuracy = accuracy_score(y_test_array, y_pred_tags_array)
    precision = [0]  # precision_score(y_test_array, y_pred_tags_array)
    recall = [0]  # recall_score(y_test_array, y_pred_tags_array)
    f1 = [0]  # f1_score(y_test_array, y_pred_tags_array)

    return accuracy, precision, recall, f1

def validation(model, loader, criterion, device, verbose=False):
    with torch.no_grad():
        val_epoch_loss = 0
        val_epoch_acc = 0
        val_epoch_precision = 0
        val_epoch_recall = 0
        val_epoch_f1 = 0

        model.eval()
        for x, y in loader:
            inputs, labels = x.to(device), y.to(device)
            if verbose:
                print(f'inputs.shape: {inputs.shape}')
            inputs = inputs.unsqueeze(1)
            if verbose:
                print(f'inputs.shape: {inputs.shape}')
            output = model(inputs)
            if verbose:
                print(f'output: {output}, shape: {output.shape}')
                print(f'labels: {labels}, shape: {labels.shape}')
            val_loss = criterion(output, labels)
            val_acc, val_precision, val_recall, val_f1 = prediction_metrics(y_pred=output, y_test=labels)

            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()
            # val_epoch_precision += val_precision.item()
            # val_epoch_recall += val_recall.item()
            # val_epoch_f1 += val_f1.item()

    return val_epoch_loss, val_epoch_acc, val_epoch_precision, val_epoch_recall, val_epoch_f1

def train(model, train_dl, val_dl, epochs, criterion, device, verbose=False):
    print("Begin training.")
    # Create dictionnaries to store the training and validation accuracy, precision, recall and loss
    loss_stats = {
        'train': [],
        "val": []
    }
    accuracy_stats = {
        'train': [],
        "val": []
    }
    precision_stats = {
        'train': [],
        "val": []
    }
    recall_stats = {
        'train': [],
        "val": []
    }
    f1_stats = {
        'train': [],
        "val": []
    }

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)    # enumerate epochs

    # initial validation
    model.eval()
    val_epoch_loss, val_epoch_acc, val_epoch_precision, val_epoch_recall, val_epoch_f1 = validation(model, val_dl, criterion, device)
    print(f'=== initialized === validation loss {val_epoch_loss/len(val_dl)} === validation accuracy {val_epoch_acc/len(val_dl)} ===')
    min_val_loss = val_epoch_loss
    torch.save(model.state_dict(), 'model.path')

    for epoch in range(epochs):  # loop over the dataset multiple times
        start = time.time()
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        train_epoch_precision = 0
        train_epoch_recall = 0
        train_epoch_f1 = 0

        model.train()
        for x_batch, y_batch in train_dl:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad() # zero the parameter gradients
            if verbose:
                print(f'inputs.shape: {inputs.shape}')
            inputs = inputs.unsqueeze(1)
            if verbose:
                print(f'inputs.shape: {inputs.shape}')
            # forward + backward + optimize
            output = model(inputs)
            if verbose:
                print(f'output: {output}, shape: {output.shape}')
                print(f'labels: {labels}, shape: {labels.shape}')

            loss = criterion(output, labels)
            acc, precision, recall, f1 = prediction_metrics(y_pred=output, y_test=labels)

            # update model weights
            loss.backward()
            optimizer.step()

            # print statistics
            train_epoch_loss += loss.item() #sum of all losses over every batch, for every batch you get 1 loss more
            train_epoch_acc += acc.item()
            # train_epoch_precision += precision.item()
            # train_epoch_recall += recall.item()
            # train_epoch_f1 += f1.item()

        # VALIDATION
        val_epoch_loss, val_epoch_acc, val_epoch_precision, val_epoch_recall, val_epoch_f1 = validation(model, val_dl, criterion, device)

        loss_stats['train'].append(train_epoch_loss / len(train_dl))
        loss_stats['val'].append(val_epoch_loss / len(val_dl))
        accuracy_stats['train'].append(train_epoch_acc / len(train_dl))
        accuracy_stats['val'].append(val_epoch_acc / len(val_dl))
        # precision_stats['train'].append(train_epoch_precision / len(train_dl))
        # precision_stats['val'].append(val_epoch_precision / len(val_dl))
        # recall_stats['train'].append(train_epoch_recall / len(train_dl))
        # recall_stats['val'].append(val_epoch_recall / len(val_dl))
        # f1_stats['train'].append(train_epoch_f1 / len(train_dl))
        # f1_stats['val'].append(val_epoch_f1 / len(val_dl))


        print(f' Epoch {epoch+1} {time.time()-start:.0f}s | '
              f'Training loss: {train_epoch_loss / len(train_dl)} |  '
              f'Validation loss: {val_epoch_loss / len(val_dl)} |  '
              f'Training accuracy: {train_epoch_acc / len(train_dl)} |  '
              f'Validation accuracy: {val_epoch_acc / len(val_dl)} |  ')
              # f'Training precision: {train_epoch_precision / len(train_dl)} |  '
              # f'Validation precision: {val_epoch_precision / len(val_dl)} |  '
              # f'Training recall: {train_epoch_recall / len(train_dl)} |  '
              # f'Validation recall: {val_epoch_recall / len(val_dl)} |  '
              # f'Training f1: {train_epoch_f1 / len(train_dl)} |  '
              # f'Validation f1: {val_epoch_f1 / len(val_dl)} |  ')

        scheduler.step(val_epoch_loss)

        if val_epoch_loss < min_val_loss:
            print(f'Validation loss decreased ({min_val_loss:.6f} --> {val_epoch_loss:.6f}).  Saving model ...')
            torch.save(model.state_dict(), f'{PATH}/model_full.path')
            min_val_loss = val_epoch_loss

    print('Finished Training')
    return model, loss_stats, accuracy_stats, precision_stats, recall_stats, f1_stats
