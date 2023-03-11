from torch.utils.data import DataLoader
from Net import *
import argparse
import os
import torch.nn as nn
from torch.optim import RMSprop, Adam
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import orjson

from sklearn import metrics
import numpy as np
from sklearn.metrics import r2_score
def mape(y_true, y_pred):
    np.seterr(divide='ignore', invalid='ignore')
    return np.mean(np.abs((y_true - y_pred) / (y_true+1e-6)))

parser = argparse.ArgumentParser(description='Chl Prediction')
parser.add_argument('--epochs', default=100, type=int, help='the train epochs of net work')
parser.add_argument('--lr', default=0.0001, type=float, help='the learning rate of the net work')
parser.add_argument('--rnn_number_layer', default=4, type=int, help='the number of the GRU/LSTM layer ')
parser.add_argument('--rnn_hidden_size', default=64, type=int, help='the hidden size of GRU or LSTM')
parser.add_argument('--encoder_input_size', default=16 * 4 + 4, type=int, help='the input size of encoder')
parser.add_argument('--encoder_hidden_size', default=64, type=int, help='the hidden size of encoder')
parser.add_argument('--decoder_output_size', default=3, type=int, help='the mlp decoder hidden size')
parser.add_argument('--output_size', default=2, type=int, help='the predict value number')
parser.add_argument('--batch_size', default=10, type=int, help='the batch size of train loader')

args = parser.parse_args()

if __name__ == '__main__':
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    net = SCM(
        encoder_input_size=args.encoder_input_size,
        num_layers=args.rnn_number_layer,
        rnn_hidden_size=args.rnn_hidden_size,
        encoder_hidden_size=args.encoder_hidden_size
    ).to(device)
    
    loss_func = nn.L1Loss().to(device)
    # optimizer = RMSprop(net.parameters(), lr=args.lr, momentum=0.99)
    optimizer = Adam(net.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    if not os.path.exists('data/data.json'):
        dataset = DatasetProcess()
        dataset.save()

    train_dataloader = DataLoader(ChlDataset(train=True), args.batch_size, shuffle=True)
    test_dataloader = DataLoader(ChlDataset(train=False), args.batch_size, shuffle=False)
    
    mean = ChlDataset(train=True).mean
    std = ChlDataset(train=True).std

    train_loss = []
    test_losss = []
    test_rmse = []
    test_mae = []
    test_r2 = []
    test_mse = []

    for epoch in range(args.epochs):
        epoch_loss = torch.zeros(1)
        for i, (data, label) in enumerate(train_dataloader):
            output = net(data.to(device))
            optimizer.zero_grad()
            loss = loss_func(output, label[:, :300].to(device))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_loss.append(float((epoch_loss / len(train_dataloader)).item()))

        scheduler.step(epoch_loss)

        print(f'Processing: [{epoch} / {args.epochs}] | Loss: {round((epoch_loss/len(train_dataloader)).item(), 6)} | Learning Rate: {optimizer.state_dict()["param_groups"][0]["lr"]}')

        if (epoch + 1) % 10 == 0:
            prediction = []
            test_loss = torch.zeros(1)
            with torch.no_grad():
                for i, (data, label) in enumerate(test_dataloader):
                    output = net(data.to(device))
                    loss = loss_func(output, label[:, :300].to(device))
                    test_loss += loss.item()

                    output = output.detach().cpu().numpy()
                    label = label.cpu().numpy()

                    output[:, :300] = output[:, :300] * std[:300] + mean[:300]
                    label[:, :300] = label[:, :300] * std[:300] + mean[:300]
                    for j in range(label.shape[0]):
                        prediction.append({'label': label[j, :300].tolist(), 'predict': output[j, ...].tolist()})

                file_name = f'Result/{epoch}.json'
                with open(file_name, 'w') as f:
                    f.write(orjson.dumps(prediction).decode())

                f.close()

                print(f'Processing: [{epoch} / {args.epochs}] | Test Loss: {round((test_loss/len(test_dataloader)).item(), 6)}')

                test_losss.append(float((test_loss / len(test_dataloader)).item()))  # 加的
                # 加评价指标
                y_true = np.array(label[:, :300])
                y_pred = np.array(output[:, :300])

                test_mse.append(metrics.mean_squared_error(y_true, y_pred))
                print('MSE:', metrics.mean_squared_error(y_true, y_pred))

                test_rmse.append(np.sqrt(metrics.mean_squared_error(y_true, y_pred)))
                print('RMSE:', np.sqrt(metrics.mean_squared_error(y_true, y_pred)))

                test_mae.append(metrics.mean_absolute_error(y_true, y_pred))
                print('MAE:', metrics.mean_absolute_error(y_true, y_pred))

                print('MAPE:', mape(y_true, y_pred))

                score = r2_score(y_true, y_pred)
                test_r2.append(score)
                print("r^2的值：", score)

    with open("Loss/train_loss.txt", 'w') as train_los:
        train_los.write(str(train_loss))

    with open("Loss/test_losss.txt", 'w') as test_los:
        test_los.write(str(test_losss))

    with open("Evaluation/test_rmse.txt", 'w') as test_rms:
        test_rms.write(str(test_rmse))

    with open("Evaluation/test_mae.txt", 'w') as test_ma:
        test_ma.write(str(test_mae))

    with open("Evaluation/test_r2.txt", 'w') as test_r:
        test_r.write(str(test_r2))
    with open("Evaluation/test_mse.txt", 'w') as test_ms:
        test_ms.write(str(test_mse))

