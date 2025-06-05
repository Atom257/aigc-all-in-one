import numpy as np
import torch
import torch.nn as nn


class TorchCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ):
        super(TorchCNN, self).__init__()
        self.layer = nn.Conv2d(in_channels, out_channels, kernel_size, bias=True)

    def forward(self, x):
        y_pred = self.layer(x)
        return y_pred


class MyCnn:
    def __init__(self, input_high, input_width, weights, kernel_size, bias):
        self.height = input_high
        self.width = input_width
        self.weights = weights
        self.kernel_size = kernel_size
        self.bias = bias

    def forward(self, x):
        output = []
        for idx, kernel_weight in enumerate(self.weights):
            print(f"kernel weight: \n{kernel_weight}")
            print(f"kernel weight shape: \n{kernel_weight.shape}")
            print('---'*8)
            kernel_weight = kernel_weight.squeeze().numpy()

            print(f"kernel weight squeeze: \n{kernel_weight}")
            print(f"kernel weight shape: \n{kernel_weight.shape}")
            print('---' * 8)


            kernel_output = np.zeros((self.height - self.kernel_size + 1, self.width - self.kernel_size + 1))
            for i in range(self.height - self.kernel_size + 1):
                for j in range(self.width - self.kernel_size + 1):
                    window = x[i:i + self.kernel_size, j:j + self.kernel_size]
                    kernel_output[i, j] = np.sum(window * kernel_weight)

            kernel_output += self.bias[idx].item()
            output.append(kernel_output)
        output = np.array(output)
        return output


x = np.array([[0.1, 0.2, 0.3, 0.4],
              [-3, -4, -5, -6],
              [5.1, 6.2, 7.3, 8.4],
              [-0.7, -0.8, -0.9, -1]])  # 网络输入

in_channels = 1
out_channels = 3
kernel_size = 2
torch_model = TorchCNN(in_channels, out_channels, kernel_size)
print(torch_model.state_dict())
torch_w = torch_model.state_dict()['layer.weight']
torch_bias = torch_model.state_dict()['layer.bias']

torch_x = torch.tensor([x], dtype=torch.float32).unsqueeze(0)  # 添加 batch 和 channel 维度

output = torch_model.forward(torch_x)
output = output.detach().numpy()
print("torch模型预测结果\n", output, output.shape)
print("---------------")

diy_model = MyCnn(x.shape[0], x.shape[1], torch_w, kernel_size, torch_bias)
diy_output = diy_model.forward(x)
print("diy模型预测结果\n", output, )

print(np.allclose(output[0], diy_output))  # 应该为 True
