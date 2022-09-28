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
    cfg,
    model,
    phases,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    dataset_sizes,
    num_epochs=1
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")

        preds_list = []
        labels_list = []

        if "train" in phases:
            train_evaluation_matrix = torch.empty((num_epochs, 4)).to(device)

        if "val" in phases:
            val_evaluation_matrix = torch.empty((num_epochs, 4)).to(device)

        # Each epoch has a training and validation phase
        for phase in phases:
            print(f'[phase]:{phase}')
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_prob = 0.0

            # Iterate over data.
            for i, (images, labels) in enumerate(dataloaders[phase]):
                images = images.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(images)
                    values, preds = torch.max(outputs, 1)
                    values = F.sigmoid(values)
                    # print(values)
                    prob = torch.sum(values)
                    # print(prob)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_prob += prob
                # print('running_prob', running_prob)
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)
                preds_list = preds_list + preds.tolist()
                labels_list = labels_list + labels.data.tolist()

            if phase == "train":
                scheduler.step()

            # evaluation calculation
            epoch_prob = running_prob / dataset_sizes[phase]
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

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

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print("-" * 100)
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)

    if phases[0] != "test":
        return model, train_evaluation_matrix, val_evaluation_matrix
    else:
        print("there is no return value becasue of test mode")
        return epoch_prob


def visualize_model(model, dataloaders, phase="val", num_images=6):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    was_training = model.training
    model.eval()
    images_so_far = 0
    # fig = plt.figure()

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloaders[phase]):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for j in range(images.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis("off")
                # ax.set_title(f'predicted: {class_names[preds[j]]}')
                plt.imshow(images.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
