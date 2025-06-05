import numpy as np
import torch
import torch.nn as nn


class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, ):
        super(TorchModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1, bias=True)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2, bias=True)

    def forward(self, x):
        hidden = self.layer1(x)
        print("hidden:", hidden)
        y_pred = self.layer2(hidden)
        return y_pred


class MyModel:
    def __init__(self, weight1, weight2, bias1, bias2):
        self.weight1 = weight1
        self.weight2 = weight2
        self.bias1 = bias1
        self.bias2 = bias2

    def forward(self, x):
        hidden = np.dot(x, self.weight1.T) + self.bias1
        y_pred = np.dot(hidden, self.weight2.T) + self.bias2

        return y_pred


x = np.array([1, 0, 0])

torch_model = TorchModel(len(x), 5, 2)
print(torch_model.state_dict(), "\n")
torch_model_weight1 = torch_model.state_dict()["layer1.weight"].numpy()
torch_model_weight2 = torch_model.state_dict()["layer2.weight"].numpy()
torch_model_bias1 = torch_model.state_dict()["layer1.bias"].numpy()
torch_model_bias2 = torch_model.state_dict()["layer2.bias"].numpy()

print(f"torch model weight1: {torch_model_weight1}\n")
print(f"torch model weight2: {torch_model_weight2}\n")
print(f"torch model bias1: {torch_model_bias1}\n")
print(f"torch model bias2: {torch_model_bias2}\n")

torch_x = torch.FloatTensor([x])
y_pred = torch_model.forward(torch_x)
print(f"torch模型预测结果：{y_pred}")

my_model = MyModel(torch_model_weight1, torch_model_weight2, torch_model_bias1, torch_model_bias2)
y_pred_my = my_model.forward(np.array([x]))

print("diy模型预测结果：", y_pred_my)
print("-----------------------------")
