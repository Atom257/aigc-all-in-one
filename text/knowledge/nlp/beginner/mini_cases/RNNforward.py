import numpy as np
import torch
import torch.nn as nn


class TorchRnn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TorchRnn, self).__init__()
        self.layer = nn.RNN(input_size, hidden_size, bias=True, batch_first=True)

    def forward(self, x):
        return self.layer(x)


class MyRnn:
    def __init__(self, w_ih, w_hh, hidden_size, ih_bias, hh_bias):
        self.w_ih = w_ih
        self.w_hh = w_hh
        self.hidden_size = hidden_size
        self.ih_bias = ih_bias
        self.hh_bias = hh_bias

    def forward(self, x):
        ht = np.zeros((self.hidden_size))
        output = []
        for xt in x:
            # print(f"xt:{xt}\n")
            ux = np.dot(self.w_ih, xt) + self.ih_bias
            # print(f"ux:{ux}\n")
            uh = np.dot(self.w_hh, ht) + self.hh_bias
            # print(f"uh:{uh}\n")
            ht_next = np.tanh(ux + uh)
            # print(f"ht_next:{ht_next}\n")
            output.append(ht_next)
            ht = ht_next
        return np.array(output), ht


x = np.array([[1, 2, 3],
              [3, 4, 5],
              [5, 6, 7]])  # 网络输入

hidden_size = 4
torch_model = TorchRnn(3, hidden_size)

print("torch_model.state_dict()\n", torch_model.state_dict())
print("---------------")

w_ih = torch_model.state_dict()['layer.weight_ih_l0'].numpy()
w_hh = torch_model.state_dict()['layer.weight_hh_l0'].numpy()
ih_bias = torch_model.state_dict()['layer.bias_ih_l0'].numpy()
hh_bias = torch_model.state_dict()['layer.bias_hh_l0'].numpy()

torch_x = torch.FloatTensor([x])
output, h = torch_model(torch_x)
print(h)
print(output.detach().numpy(), "torch模型预测结果")
print(h.detach().numpy(), "torch模型预测隐含层结果")
print("---------------")

my_model = MyRnn(w_ih, w_hh, hidden_size, ih_bias, hh_bias)
my_output, my_h = my_model.forward(x)
print(output, "diy模型预测结果")
print(my_h, "diy模型预测隐含层结果")

print(np.allclose(output.detach().numpy(), my_output), "模型预测结果是否相等")
