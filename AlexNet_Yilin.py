import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os


if __name__ == '__main__':

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using CUDA")

    t1=time.time()

    image_dir = 'food-101/images'
    image_size = (224, 224)
    batch_size = 32 #64 16   #4
    epochs = 10
    cls=101   #number of classes

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_data = torchvision.datasets.ImageFolder(root='food-101/images', transform=transform)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)

    test_data = torchvision.datasets.ImageFolder(root='food-101/testing', transform=transform) #train=False, download=True,
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)

    with open("food-101/meta/classes.txt") as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    classes = [x.strip() for x in content]

    # Load the pretrained model from pytorch
    # AlexNet_model  = models.alexnet()   #poor accuracy
    AlexNet_model = models.alexnet(pretrained=True)

    #Model description
    # print(AlexNet_model.eval())

    AlexNet_model.classifier[4] = nn.Linear(4096,1024)
    AlexNet_model.classifier[6] = nn.Linear(1024,101)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    AlexNet_model.to(device)

    #Loss
    criterion = nn.CrossEntropyLoss()

    #Optimizer(SGD)
    optimizer = optim.SGD(AlexNet_model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.Adam(AlexNet_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    train_loss_history = []
    train_acc_history = []
    valid_loss_history = []
    valid_acc_history = []

    def trainval(trainloader, optimizer, testloader):
        # Training
        AlexNet_model.train()
        running_loss = 0.0
        loss1 = 0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            # forward + backward + optimize
            output = AlexNet_model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            loss1 += loss.item()
            # running_loss += loss.item()
            # if i % 500 == 499:    # print every 500 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #           (ep + 1, i + 1, running_loss / 500))
            #     running_loss = 0.0

            _, preds = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            del inputs, labels, output, preds
            torch.cuda.empty_cache()

        loss_tr = loss1 / len(trainloader)
        acc_tr = 100 * correct / total

        #evaluation
        AlexNet_model.eval()
        with torch.no_grad():
            running_loss = 0.0
            loss1 = 0
            correct = 0
            total = 0

            for i, data in enumerate(testloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                output = AlexNet_model(inputs)
                loss = criterion(output, labels)
                loss1 += loss.item()

                # running_loss += loss.item()
                # if i % 500 == 499:    # print every 500 mini-batches
                #     print('[%d, %5d] loss: %.3f' %
                #           (ep + 1, i + 1, running_loss / 500))
                #     running_loss = 0.0

                _, preds = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

                del inputs, labels, output, preds
                torch.cuda.empty_cache()

        loss_val = loss1 / len(trainloader)
        acc_val = 100 * correct / total

        return loss_tr, acc_tr, loss_val, acc_val


    for ep in range(epochs):  # loop over the dataset multiple times
        t2 = time.time()
        loss_tr, acc_tr, loss_val, acc_val = trainval(trainloader, optimizer, testloader)
        print('time epoch' + str(ep+1) + ': ' + str(time.time()-t2))  #every epoch took time
        print("loss train: " + str(loss_tr))
        print("accuaracy train %: " + str(acc_tr))
        print("loss val: " + str(loss_val))
        print("accuaracy val %: " + str(acc_val))
        print("-------------------------------------")

        train_loss_history.append(loss_tr)
        train_acc_history.append(acc_tr)
        valid_loss_history.append(loss_val)
        valid_acc_history.append(acc_val)

    t3 = time.time()
    print("Finished Training of AlexNet, time: " + str(t3-t1))
    print(train_loss_history)
    print(train_acc_history)
    print(valid_loss_history)
    print(valid_acc_history)
    print("time total: " + str(time.time()-t1))

    x = np.arange(1, 11)
    plt.suptitle("Implementing AlexNet on Food-101 Dataset", size=16)
    plt.subplot(1, 2, 1)
    plt.plot(x, train_loss_history)
    plt.plot(x, valid_loss_history)
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train', 'valid'])

    plt.subplot(1, 2, 2)
    plt.plot(x, train_acc_history)
    plt.plot(x, valid_acc_history)
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'valid'])
    plt.show()


# Resources:
# https://analyticsindiamag.com/implementing-alexnet-using-pytorch-as-a-transfer-learning-model-in-multi-class-classification/
# https://pytorch.org/vision/stable/models.html
# https://www.kaggle.com/theimgclist/multiclass-food-classification-using-tensorflow





