import torch

#1.准备数据集
x_data = torch.Tensor([[1.0],[2.0],[3.0],[4.0],[5.0]])
y_data = torch.Tensor([[0],[0],[0],[1],[1]])

#2.设计网络模型
class LogisticRegressionModel(torch.nn.Module):
    #初始化
    def __init__(self):
        super(LogisticRegressionModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)
        
    def forward(self,x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
model = LogisticRegressionModel()

#3.构建损失函数和优化器的选择
criterion = torch.nn.BCELoss(size_average=False)  #BCE 交叉熵
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
#训练
for epoch in range(2000):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch,loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
#测试模型
x_test = torch.Tensor([6])
y_test = model(x_test)
print('y_pred = ',y_test.data)

