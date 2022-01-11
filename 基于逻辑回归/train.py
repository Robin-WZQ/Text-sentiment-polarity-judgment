import torch

def train(device,train_iter,optimizor,criterion,n_epoches,batch_size,net,test_iter):
    print("training on :{}".format(device))
    for epoch in range(n_epoches):
        acc = 0.0
        n=0.0
        for i,data in enumerate(train_iter):
            X,Y=data[0],data[1]
            predict = net(X)
            predict = predict.view(1,batch_size)
            Y = Y.unsqueeze(0)
            # print(X.shape,predict,predict.shape,Y,Y.shape)
            # print(predict.shape,Y.shape)
            Loss = criterion(predict.float(),Y.float())

            optimizor.zero_grad()
            Loss.backward()
            optimizor.step()

            for k in range(len(Y)):
                n+=1
                if predict[0][k]>0.5 and Y[0][k]==1:
                    acc+=1
                if predict[0][k]<0.5 and Y[0][k]==0:
                    acc+=1
            
        acc = acc/n

        # print(predict)
        test_accuracy = evaluate(test_iter,net,1)
        if (epoch+1) % 10 == 0:
            print("epoch {:0}--------  Loss={:1} train_acc={:2} test_acc={:3}".format(epoch,Loss.item(),acc,test_accuracy))
            torch.save(net.state_dict(), 'results/'+f'{epoch}_logistic_net.pkl')
    return 0

def evaluate(test_iter,net,batch_size):
    acc_sum, n = 0.0, 0.0
    for i,data in enumerate(test_iter):
        X,Y=data[0],data[1]         
        predict = net(X)
        predict = predict.view(1,batch_size)
        Y = Y.unsqueeze(0)
        for j in range(len(Y)):
            if(Y[0][j]==predict[0][j]):
                acc_sum+=1
            n+=1
    return acc_sum / n    