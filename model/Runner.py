import copy
import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from external_library import MTCNN


def macro_evaluation(preds_list, labels_list):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch_preds = torch.Tensor(preds_list).int().to(device)
    torch_labels = torch.Tensor(labels_list).int().to(device)

    num_classes = len(torch.unique(torch_labels))
    precision = 0
    recall = 0

    for class_ in torch.unique(torch_labels):
        temp_true = torch.where(torch_labels == class_, 1, 0)
        temp_pred = torch.where(torch_preds == class_, 1, 0)

        tp = torch.where((temp_true == 1) & (temp_pred == 1), 1, 0).sum()
        fp = torch.where((temp_true == 0) & (temp_pred == 1), 1, 0).sum()
        fn = torch.where((temp_true == 1) & (temp_pred == 0), 1, 0).sum()

        temp_recall = tp / (tp + fn + 1e-6)
        temp_precision = tp / (tp + fp + 1e-6)

        precision += temp_precision
        recall += temp_recall

    precision /= num_classes
    recall /= num_classes

    return recall, precision


def runner(
    model,
    phases,
    loss_fn,
    optimizer,
    scheduler,
    dataloaders,
    num_epochs=1
):
    # dataset_size = len(dataloader.dataset)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    
    if "train" in phases:
        train_evaluation_matrix = torch.empty((num_epochs, 4)).to(device)

    if "val" in phases:
        val_evaluation_matrix = torch.empty((num_epochs, 4)).to(device)
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
    
        for phase in phases:
            # print(f'[phase]:{phase}')
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            
            dataset_size = len(dataloaders[phase].dataset)
            preds_list = []
            labels_list = []

            running_loss = 0.0
            running_corrects = 0
            running_prob = 0.0
            
            # Iterate over data.
            for i, (embs, labels) in enumerate(dataloaders[phase]):
                embs = embs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(embs)
                    values, preds = torch.max(outputs, 1)
                    # values = F.sigmoid(values)
                    # print(values)
                    # print(preds)
                    prob = torch.sum(values)
                    # print(prob)
                    # print(labels.shape, preds.shape)
                    loss = loss_fn(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_prob += prob
                # print('running_prob', running_prob)
                running_loss += loss.item() * embs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                preds_list = preds_list + preds.tolist()
                labels_list = labels_list + labels.data.tolist()

            if phase == "train":
                scheduler.step()

            # evaluation calculation
            epoch_prob = running_prob / dataset_size
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            epoch_recall, epoch_precision = macro_evaluation(preds_list,
                                                             labels_list)

            # evaluation print
            msg = (
                f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} "
                f"recall: {epoch_recall:.4f} Precision: {epoch_precision:.4f} "
                f'avg_prob: {epoch_prob:.4f}'
            )
            
            print(msg)

            # save evaluation results
            if phase == "train":
                train_evaluation_matrix[epoch][0] = epoch_loss
                train_evaluation_matrix[epoch][1] = epoch_acc
                train_evaluation_matrix[epoch][2] = epoch_recall
                train_evaluation_matrix[epoch][3] = epoch_precision
            elif phase == "val":
                val_evaluation_matrix[epoch][0] = epoch_loss
                val_evaluation_matrix[epoch][1] = epoch_acc
                val_evaluation_matrix[epoch][2] = epoch_recall
                val_evaluation_matrix[epoch][3] = epoch_precision
            
    print("-" * 70)
        
    time_elapsed = time.time() - since
    msg = (f'Training complete in {time_elapsed // 60:.0f}m '
           f'{time_elapsed % 60:.0f}s')
    print(msg)

    if phases[0] == "test":
        print("there is no return value becasue of test mode")
    else:
        return model, train_evaluation_matrix, val_evaluation_matrix