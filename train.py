import torch
import torch.nn.functional as F
from torch.autograd import Variable

from backbone import Network


class Dataset:
    def __init__(self):
        # 必须要使用unsqueeze,否则会报错
        # 神经网络的记忆是有一定的范围的，如果让现在的神经网络拟合的数据增加10倍，现在的神经网络就不能拟合。而现在的神经网络却能够把下面的数据拟合得很好。
        x = torch.unsqueeze(torch.linspace(0, 10, 11), dim=1)
        self.x = Variable(torch.unsqueeze(x, dim=1))
        self.y = Variable(self.x.pow(2))
        self.x_test = x + 0.5
        self.y_test = self.x_test.pow(2)
        self.net = Network()

    def train(self):
        loss_function = F.mse_loss
        optimizer = torch.optim.SGD(self.net.parameters(), lr=0.001)
        for t in range(10000):
            prediction = self.net(self.x)
            loss = loss_function(prediction, self.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def test(self):
        self.net.eval()
        with torch.no_grad():
            for i in range(self.x_test.size()[0]):
                predict_y = self.net(self.x_test[i])
                real_y = self.y_test[i]
                print(i, predict_y, real_y, predict_y - real_y)


if __name__ == '__main__':
    dataset = Dataset()
    dataset.train()
    dataset.test()
