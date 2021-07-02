from data_loader import *
from model import *
import pickle
import numpy as np

import torch
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import time
import random

matplotlib.use('Agg')

def main(task, kind, n_epoch):
    loader = GPGenerator(batch_size=16, num_classes=1, data_source='gp', is_train=True)
    
    model = NPregression(kind=kind, dim_input=1, dim_output=1).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    KL_list, NLL_list = [], []
    train_loss_list, test_loss_list = [], []
    train_MSE_list, test_MSE_list = [], []
    for epoch in range(n_epoch):
        model.train()
        
        (Cx, Tx), (Cy, Ty), _, _ = loader.generate_mixture_batch(is_test=False)    
            
        Cx = torch.tensor(Cx, dtype=torch.float)
        Cy = torch.tensor(Cy, dtype=torch.float)
        Tx = torch.tensor(Tx, dtype=torch.float)
        Ty = torch.tensor(Ty, dtype=torch.float)
        
        _, train_KL, train_NLL, train_MSE = model(Cx.to(device), Cy.to(device), Tx.to(device), Ty.to(device))
        train_loss = train_KL + train_NLL
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
                    
        KL_list.append(train_KL.item())
        NLL_list.append(train_NLL.item())
        train_loss_list.append(train_loss.item())
        train_MSE_list.append(train_MSE.item())
        
        with torch.no_grad():
            model.eval()
        
            (Cx, Tx), (Cy, Ty), _, _ = loader.generate_mixture_batch(is_test=False)    
        
            Cx = torch.tensor(Cx, dtype=torch.float)
            Cy = torch.tensor(Cy, dtype=torch.float)
            Tx = torch.tensor(Tx, dtype=torch.float)
            Ty = torch.tensor(Ty, dtype=torch.float)
        
            _, _, test_NLL, test_MSE = model(Cx.to(device), Cy.to(device), Tx.to(device), Ty.to(device))
            test_loss = test_NLL
        
            test_loss_list.append(test_loss.item())
            test_MSE_list.append(test_MSE.item())
            
        if (epoch+1) % 10000 == 0:
            print('[Epoch %d] KL: %.3f, NLL: %.3f, train_loss: %.3f, test_loss: %.3f, train_MSE: %.3f, test_MSE: %.3f' 
                  % (epoch+1, train_KL, train_NLL, train_loss, test_loss, train_MSE, test_MSE))
            torch.save(model.state_dict(), 'models/'+task+kind+'/'+str(epoch+1)+'.pt')
            
            torch.save(KL_list, 'loss/'+task+kind+'/KL_list.pt')
            torch.save(NLL_list, 'loss/'+task+kind+'/NLL_list.pt')
            torch.save(train_loss_list, 'loss/'+task+kind+'/train_loss_list.pt')
            torch.save(test_loss_list, 'loss/'+task+kind+'/test_loss_list.pt')
            torch.save(train_MSE_list, 'loss/'+task+kind+'/train_MSE_list.pt')
            torch.save(test_MSE_list, 'loss/'+task+kind+'/test_MSE_list.pt')
        
            
def regression(task, kind):
    model = NPregression(kind=kind, dim_input=1, dim_output=1).to(device)
    model.load_state_dict(torch.load('models/'+task+kind+'/1000000.pt'))
    model.eval()
    result_path = 'plots/'+task+kind
    
    loader = GPGenerator(batch_size=2, num_classes=1, data_source='gp', is_train=False)
    
    (Cx, Tx), (Cy, Ty), _, _ = loader.generate_mixture_batch(is_test=True)    
    
    Cx = torch.tensor(Cx, dtype=torch.float)
    Cy = torch.tensor(Cy, dtype=torch.float)
    Tx = torch.tensor(Tx, dtype=torch.float)
    Ty = torch.tensor(Ty, dtype=torch.float)

    y_dist, _, _, MSE = model(Cx.to(device), Cy.to(device), Tx.to(device), Ty.to(device))
           
    Cx, Cy = torch.squeeze(Cx, -1).cpu(), torch.squeeze(Cy, -1).cpu()
    Tx, Ty = torch.squeeze(Tx, -1).cpu(), torch.squeeze(Ty, -1).cpu()
    
    mean, std = y_dist.mean.detach(), y_dist.stddev.detach()
    mean, std = torch.squeeze(mean, -1).cpu(), torch.squeeze(std, -1).cpu()
    
    plt.figure()
    plt.title('regression')
    plt.plot(Tx[0], Ty[0], 'k:', label='True')
    plt.plot(Cx[0], Cy[0], 'k^', markersize=10, label='Contexts')
    plt.plot(Tx[0], mean[0], 'b', label='Predictions')
    plt.fill(torch.cat((Tx[0], torch.flip(Tx[0], [0])),0),
             torch.cat((mean[0] - 1.96 * std[0], torch.flip(mean[0] + 1.96 * std[0], [0])),0),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(min(Ty[0]) - 0.3 * (max(Ty[0]) - min(Ty[0])), max(Ty[0]) + 0.3 * (max(Ty[0]) - min(Ty[0])))
    plt.savefig(result_path+'.png')
    plt.close()

    
    
def evaluation(task, kind, NLL_tilt_w, lamb):
    model = NPregression(kind=kind, dim_input=1, dim_output=1).to(device)
    model.load_state_dict(torch.load('models/'+task+kind+'/1300000.pt'), strict=False)
    model.eval()
    
    print(1)
    loader = GPGenerator(batch_size=4000, num_classes=1, data_source='gp', is_train=False)
    (Cx, Tx), (Cy, Ty), _, _ = loader.generate_mixture_batch(is_test=True)    
        
    Cx = torch.tensor(Cx, dtype=torch.float)
    Cy = torch.tensor(Cy, dtype=torch.float)
    Tx = torch.tensor(Tx, dtype=torch.float)
    Ty = torch.tensor(Ty, dtype=torch.float)
        
    y_dist, _, _, MSE = model(Cx.to(device), Cy.to(device), Tx.to(device), Ty.to(device), NLL_tilt_w=NLL_tilt_w, lamb=lamb)
    
    NLL_tensor = -y_dist.log_prob(Ty.to(device)).sum(-1).mean(-1)
    
    mean = y_dist.mean.detach().cpu()        
    MSE_tensor = ((Ty-mean)**2).squeeze(-1).mean(-1)     
    print('NLL', NLL_tensor.mean(), 'MSE', MSE_tensor.mean())
    

def count_parameters():
    model = NPregression(kind=kind, dim_input=1, dim_output=1).to(device)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    
if __name__ == '__main__':  
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    n_epoch = 1000000

    task = 'gp'
    kind = 'NP'
    
    tic = time.time()
    main(task, kind, n_epoch)
    toc = time.time()
    
    mon, sec = divmod(toc-tic, 60)
    hr, mon = divmod(mon, 60)
    print('total wall-clock time is ', int(hr),':',int(mon),':',int(sec))