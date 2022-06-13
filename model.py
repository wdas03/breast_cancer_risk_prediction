import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
#sfrom torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import roc_auc_score

def do_test(model, test_loader, device):
    '''Avg loss and AUC for an entire data loader
    '''
    model.eval()
    subject_list, exam_list = [], []
    pred_list, label_list, machine_list = [], [], []
    age_list, race_list, bmi_list = [], [], []
    birads_list, libra_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            bat_X_1 = batch['images']['same-cc'].float().to(device)
            bat_X_2 = batch['images']['same-mlo'].float().to(device)
            bat_X_3 = batch['images']['opposite-cc'].float().to(device)
            bat_X_4 = batch['images']['opposite-mlo'].float().to(device)
            bat_y = batch['label'].long().to(device)
            bat_logit_1 = model(bat_X_1)[:, :2, 1]
            bat_logit_2 = model(bat_X_2)[:, :2, 1]
            bat_logit_3 = model(bat_X_3)[:, :2, 1]
            bat_logit_4 = model(bat_X_4)[:, :2, 1]
            bat_p_1 = F.softmax(bat_logit_1)[:, 1]
            bat_p_2 = F.softmax(bat_logit_2)[:, 1]
            bat_p_3 = F.softmax(bat_logit_3)[:, 1]
            bat_p_4 = F.softmax(bat_logit_4)[:, 1]
            bat_p = torch.stack(
                [bat_p_1, bat_p_2, bat_p_3, bat_p_4], 
                dim=1)
            subject_list.append(batch['subject'])
            exam_list.append(batch['exam'])
            pred_list.append(bat_p)
            label_list.append(batch['label'])            
            machine_list.append(batch['machine'])
            age_list.append(batch['age'])
            race_list.append(batch['race'])
            bmi_list.append(batch['bmi'])
            birads_list.append(batch['birads'])
            libra_list.append(batch['libra'])            
    return(subject_list, exam_list, pred_list, label_list, 
           machine_list, age_list, race_list, bmi_list, 
           birads_list, libra_list)

def test_all_views(models, test_loader, device):
    '''Avg loss and AUC for an entire data loader
    '''
    models['same-cc'].eval()
    models['same-mlo'].eval()
    models['opp-cc'].eval()
    models['opp-mlo'].eval()
    
    subject_list, exam_list = [], []
    pred_list, label_list, machine_list = [], [], []
    age_list, race_list, bmi_list = [], [], []
    birads_list, libra_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            bat_X_1 = batch['images']['same-cc'].float().to(device)
            bat_X_2 = batch['images']['same-mlo'].float().to(device)
            bat_X_3 = batch['images']['opposite-cc'].float().to(device)
            bat_X_4 = batch['images']['opposite-mlo'].float().to(device)
            bat_y = batch['label'].long().to(device)
            
            bat_logit_1 = models['same-cc'](bat_X_1)[:, :2, 1]
            bat_logit_2 = models['same-mlo'](bat_X_2)[:, :2, 1]
            bat_logit_3 = models['opp-cc'](bat_X_3)[:, :2, 1]
            bat_logit_4 = models['opp-mlo'](bat_X_4)[:, :2, 1]
            
            bat_p_1 = F.softmax(bat_logit_1)[:, 1]
            bat_p_2 = F.softmax(bat_logit_2)[:, 1]
            bat_p_3 = F.softmax(bat_logit_3)[:, 1]
            bat_p_4 = F.softmax(bat_logit_4)[:, 1]
            
            bat_p = torch.stack(
                [bat_p_1, bat_p_2, bat_p_3, bat_p_4], 
                dim=1)
            subject_list.append(batch['subject'])
            exam_list.append(batch['exam'])
            pred_list.append(bat_p)
            label_list.append(batch['label'])            
            machine_list.append(batch['machine'])
            age_list.append(batch['age'])
            race_list.append(batch['race'])
            bmi_list.append(batch['bmi'])
            birads_list.append(batch['birads'])
            libra_list.append(batch['libra'])            
    return(subject_list, exam_list, pred_list, label_list, 
           machine_list, age_list, race_list, bmi_list, 
           birads_list, libra_list)

def test_auc_all_views(models, test_loader, device):
    '''AUC based on max prediction of 4 views
    '''
    models['same-cc'].eval()
    models['same-mlo'].eval()
    models['opp-cc'].eval()
    models['opp-mlo'].eval()
    
    y_pool, p_pool = [], []
    with torch.no_grad():
        for batch in test_loader:
            bat_X_1 = batch['images']['same-cc'].float().to(device)
            bat_X_2 = batch['images']['same-mlo'].float().to(device)
            bat_X_3 = batch['images']['opposite-cc'].float().to(device)
            bat_X_4 = batch['images']['opposite-mlo'].float().to(device)
            bat_y = batch['label'].long().to(device)
            #bat_logit_1, _, _ = model(bat_X_1)
            #bat_logit_2, _, _ = model(bat_X_2)
            #bat_logit_3, _, _ = model(bat_X_3)
            #bat_logit_4, _, _ = model(bat_X_4)
            
            bat_logit_1 = models['same-cc'](bat_X_1)[:, :2, 1]
            bat_logit_2 = models['same-mlo'](bat_X_2)[:, :2, 1]
            bat_logit_3 = models['opp-cc'](bat_X_3)[:, :2, 1]
            bat_logit_4 = models['opp-mlo'](bat_X_4)[:, :2, 1]
            
            bat_p_1 = F.softmax(bat_logit_1)[:, 1]
            bat_p_2 = F.softmax(bat_logit_2)[:, 1]
            bat_p_3 = F.softmax(bat_logit_3)[:, 1]
            bat_p_4 = F.softmax(bat_logit_4)[:, 1]
            bat_p = torch.stack(
                [bat_p_1, bat_p_2, bat_p_3, bat_p_4], 
                dim=1)
            p_pool.append(bat_p)
            y_pool.append(bat_y)
            
        ys = torch.cat(y_pool)
        ps = torch.cat(p_pool)
        
        max_ps = ps.max(1).values
        mean_ps = ps.mean(1)
        
        ys = ys.cpu().detach().numpy()
        max_ps = max_ps.cpu().detach().numpy()
        mean_ps = mean_ps.cpu().detach().numpy()
        
        test_auc_max = roc_auc_score(ys, max_ps)
        test_auc_mean = roc_auc_score(ys, mean_ps)
        
        return test_auc_max, test_auc_mean

def val_loss_all_views(models, val_loader, device, return_auc=False):
    model_same_cc = models[0] 
    model_same_mlo = models[1]
    model_opp_cc = models[2]
    model_same_mlo = models[3]
    
    '''Avg loss and AUC for an entire data loader
    '''
    criterion_ = nn.NLLLoss(reduction='sum').to(device)
    
    model_same_cc.eval()
    model_same_mlo.eval()
    model_opposite_cc.eval()
    model_opposite_mlo.eval()
    
    val_loss, n = 0, 0
    y_pool, p_pool = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch_1 = batch['images']['same-cc'].float().to(device)
            batch_2 = batch['images']['same-mlo'].float().to(device)
            batch_3 = batch['images']['opposite-cc'].float().to(device)
            batch_4 = batch['images']['opposite-mlo'].float().to(device)
            
            #print(bat_X_1.shape)
            #ii = cv2.imread(bat_X_1)
            #print(ii.shape)
            #bat_X_1 = torch.tensor(rgb2gray(bat_X_1.cpu().detach().numpy().reshape(2048, 1664, 3)).reshape(-1, 1, 2048, 1664)).float().to(device)
            #bat_X_2 = torch.tensor(rgb2gray(bat_X_2.cpu().detach().numpy().reshape(2048, 1664, 3)).reshape(-1, 1, 2048, 1664)).float().to(device)
            #print(bat_X_1.shape)
            
            
            batch_1_pred = F.log_softmax(model_same_cc(batch_1)[:, :2, 1])
            batch_2_pred = F.log_softmax(model_same_mlo(batch_2)[:, :2, 1])
            batch_3_pred = F.log_softmax(model_opp_cc(batch_3)[:, :2, 1])
            batch_4_pred = F.log_softmax(model_opp_mlo(batch_4)[:, :2, 1])
            
            n += len(batch_1)
            n += len(batch_2)
            n += len(batch_3)
            n += len(batch_4)
            
            loss_1 = criterion(batch_1_pred, batch_y)
            loss_2 = criterion(batch_2_pred, batch_y)
            loss_3 = criterion(batch_3_pred, batch_y)
            loss_4 = criterion(batch_4_pred, batch_y)
            
            val_loss += loss_1.item()
            val_loss += loss_2.item()
            val_loss += loss_3.item()
            val_loss += loss_4.item()

            
            y_pool.append(bat_y)
            y_pool.append(bat_y)
            y_pool.append(bat_y)
            y_pool.append(bat_y)# same label, different view.
            
            p_pool.append(torch.exp(batch_1_pred))
            p_pool.append(torch.exp(batch_2_pred))
            p_pool.append(torch.exp(batch_3_pred))
            p_pool.append(torch.exp(batch_4_pred))
        
            
        if return_auc:
            ys = torch.cat(y_pool)
            ps = torch.cat(p_pool)
            ys = ys.cpu().detach().numpy()
            ps = ps.cpu().detach().numpy()
            val_auc = roc_auc_score(ys, ps[:, 1])
            
            auc = val_loss/n, val_auc
            
            return auc
        
    return val_loss/n

def val_loss_single_view(model, val_loader, view, device, return_auc=True):
    '''Avg loss and AUC for an entire data loader
    '''
    criterion_ = nn.NLLLoss(reduction='sum').to(device)
    model.eval()
    val_loss, n = 0, 0
    y_pool, p_pool = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch_img = batch['images'][view].float().to(device)

            batch_y = batch['label'].long().to(device)
            
            batch_pred = F.log_softmax(model(batch_img)[:, :2, 1])
            
            n += len(batch_img)
            
            loss = criterion_(batch_pred, batch_y)
            
            val_loss += loss.item()

            
            y_pool.append(batch_y)
            p_pool.append(torch.exp(batch_pred))
        
            
        if return_auc:
            ys = torch.cat(y_pool)
            ps = torch.cat(p_pool)
            ys = ys.cpu().detach().numpy()
            ps = ps.cpu().detach().numpy()
            val_auc = roc_auc_score(ys, ps[:, 1])
            
            auc = val_loss/n, val_auc
            
            return auc
        
    return val_loss/n

def train_all_views(models, train_loader, val_loader, fold_num, device, 
          epochs=3, lr=1e-6, weight_decay=1e-4, 
          check_iters=30, log_name=None):
    
    model_rcc = models['rcc']
    model_rmlo = models['rmlo']
    model_lmlo = models['lmlo']
    
    criterion = nn.NLLLoss(reduction='mean').to(device)
    
    # Optimizers
    optimizer_rcc = Adam(model_rcc.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_rmlo = Adam(model_rmlo.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_lmlo = Adam(model_lmlo.parameters(), lr=lr, weight_decay=weight_decay)
    
    writer = open(log_name, "a+") if log_name is not None else None
    
    train_loss = 0
    best_auc_rcc = .0
    best_auc_rmlo = .0
    best_auc_lmlo = .0
    
    for i in range(epochs):
        for j, batch in enumerate(train_loader):
            model_rcc.train()
            model_rmlo.train()
            model_lmlo.train()
            
            optimizer_rcc.zero_grad()
            optimizer_rmlo.zero_grad()
            optimizer_lmlo.zero_grad()
            
            batch_1 = batch['images']['same-cc'].float().to(device)
            batch_2 = batch['images']['same-mlo'].float().to(device)
            batch_3 = batch['images']['opposite-cc'].float().to(device)
            batch_4 = batch['images']['opposite-mlo'].float().to(device)
            
            batch_y = batch['label'].long().to(device)
            
            batch_1_pred = F.log_softmax(model_same_cc(batch_1)[:, :2, 1])
            batch_2_pred = F.log_softmax(model_same_mlo(batch_2)[:, :2, 1])
            batch_3_pred = F.log_softmax(model_opp_cc(batch_3)[:, :2, 1])
            batch_4_pred = F.log_softmax(model_opp_mlo(batch_4)[:, :2, 1])
            
            loss_1 = criterion(batch_1_pred, batch_y)
            loss_1.backward()
            optimizer_same_cc.step()
            
            loss_2 = criterion(batch_2_pred, batch_y)
            loss_2.backward()
            optimizer_same_mlo.step()
            
            loss_3 = criterion(batch_3_pred, batch_y)
            loss_3.backward()
            optimizer_opp_cc.step()
            
            loss_4 = criterion(batch_4_pred, batch_y)
            loss_4.backward()
            optimizer_opp_mlo.step()
            
            train_loss += (loss_1 + loss_2 + loss_3 + loss_4) / 4
            
            total_iters = i*len(train_loader) + j + 1
            
            if total_iters%check_iters == 0:
                
                avg_val_loss_same_cc, val_auc_same_cc = val_loss_single_view(model_same_cc, val_loader, 'same-cc', device, return_auc=True)
                avg_val_loss_same_mlo, val_auc_same_mlo = val_loss_single_view(model_same_mlo, val_loader, 'same-mlo', device,  return_auc=True)
                avg_val_loss_opp_cc, val_auc_opp_cc = val_loss_single_view(model_opp_cc, val_loader, 'opposite-cc', device,  return_auc=True)
                avg_val_loss_opp_mlo, val_auc_opp_mlo = val_loss_single_view(model_opp_mlo, val_loader, 'opposite-mlo', device,  return_auc=True)
                
                avg_train_loss = train_loss/check_iters
                
                print("Iter={}, avg train loss={:.3f}, \n"
                      "\tAvg Val Loss: same-cc={:.3f}, AUC: same-cc={:.3f} \n"
                      "\tAvg Val Loss: same-mlo={:.3f}, AUC: same-mlo={:.3f} \n"
                      "\tAvg Val Loss opp-cc={:.3f}, AUC: opp-cc={:.3f} \n"
                      "\tAvg Val Loss: opp-mlo={:.3f}, AUC: opp-mlo={:.3f}".format(
                       total_iters, avg_train_loss, 
                       avg_val_loss_same_cc, val_auc_same_cc,
                       avg_val_loss_same_mlo, val_auc_same_mlo,
                       avg_val_loss_opp_cc, val_auc_opp_cc,
                       avg_val_loss_opp_mlo, val_auc_opp_mlo))
                
                """
                writer.write("Iter={}, avg train loss={:.3f}, "
                      "avg val loss={:.3f}, auc={:.3f}".format(
                    total_iters, avg_train_loss, avg_val_loss, val_auc) + "\n")
                """
                
                #scheduler.step(avg_val_loss)
                
                if val_auc_same_cc > best_auc_same_cc:
                    best_auc_same_cc = val_auc_same_cc
                    torch.save(model_same_cc.state_dict(), 'best_model_{}_same_cc.pt'.format(fold_num))
                    print("Best same-cc model saved.")
                    
                if val_auc_same_mlo > best_auc_same_mlo:
                    best_auc_same_mlo = val_auc_same_mlo
                    torch.save(model_same_mlo.state_dict(), 'best_model_{}_same_mlo.pt'.format(fold_num))
                    print("Best same-mlo model saved.")
                    
                if val_auc_opp_cc > best_auc_opp_cc:
                    best_auc_opp_cc = val_auc_opp_cc
                    torch.save(model_opp_cc.state_dict(), 'best_model_{}_opp_cc.pt'.format(fold_num))
                    print("Best opp-cc model saved.")
                    
                if val_auc_opp_mlo > best_auc_opp_mlo:
                    best_auc_opp_mlo = val_auc_opp_mlo
                    torch.save(model_opp_mlo.state_dict(), 'best_model_{}_opp_mlo.pt'.format(fold_num))
                    print("Best opp-mlo model saved.")
                
                    
                #if writer is not None:
                    #writer.add_scalar('Loss/train', avg_train_loss, total_iters)
                    #writer.add_scalar('Loss/val', avg_val_loss, total_iters)
                train_loss = 0
                
                writer.flush()
    
    model_same_cc.load_state_dict(torch.load('best_model_{}_same_cc.pt'.format(fold_num)))
    model_same_mlo.load_state_dict(torch.load('best_model_{}_same_mlo.pt'.format(fold_num)))
    model_opp_cc.load_state_dict(torch.load('best_model_{}_opp_cc.pt'.format(fold_num)))
    model_opp_mlo.load_state_dict(torch.load('best_model_{}_opp_mlo.pt'.format(fold_num)))
    
    print("Best models loaded.")
    

def train(model, train_loader, val_loader, best_name, device, 
          epochs=3, lr=1e-6, weight_decay=1e-4, 
          check_iters=30, log_name=None):
    
    criterion = nn.NLLLoss(reduction='mean').to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    writer = open(log_name, "a+") if log_name is not None else None
    train_loss = 0
    best_auc = .0
    for i in range(epochs):
        for j, batch in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            # Each iteration is equivalent to 2*batch_size.
            bat_X_1 = batch['images']['same-cc'].float().to(device)
            bat_X_2 = batch['images']['same-mlo'].float().to(device)
            bat_y = batch['label'].long().to(device)
            # forward-backward CC view images.
            bat_logit_1 = model(bat_X_1)[:, :2, 1]
            #print(bat_logit_1)
            
            bat_logp_1 = F.log_softmax(bat_logit_1)
            loss = criterion(bat_logp_1, bat_y)
            loss.backward()
            train_loss += loss.item()/2
            # forward-backward MLO view images.
            bat_logit_2 = model(bat_X_2)[:, :2, 1]
            bat_logp_2 = F.log_softmax(bat_logit_2)
            loss = criterion(bat_logp_2, bat_y)
            loss.backward()
            train_loss += loss.item()/2
            optimizer.step()
            total_iters = i*len(train_loader) + j + 1
            if total_iters%check_iters == 0:
                avg_val_loss, val_auc = val_loss(model, val_loader, device, return_auc=True)
                avg_train_loss = train_loss/check_iters
                
                print("Iter={}, avg train loss={:.3f}, "
                      "avg val loss={:.3f}, auc={:.3f}".format(
                    total_iters, avg_train_loss, avg_val_loss, val_auc))
                
                writer.write("Iter={}, avg train loss={:.3f}, "
                      "avg val loss={:.3f}, auc={:.3f}".format(
                    total_iters, avg_train_loss, avg_val_loss, val_auc) + "\n")
                
                #scheduler.step(avg_val_loss)
                
                if val_auc > best_auc:
                    best_auc = val_auc
                    torch.save(model.state_dict(), best_name)
                    print("Best model saved.")
                    
                #if writer is not None:
                    #writer.add_scalar('Loss/train', avg_train_loss, total_iters)
                    #writer.add_scalar('Loss/val', avg_val_loss, total_iters)
                train_loss = 0
                
                writer.flush()
#                 break
    print("Best model loaded.")
    model.load_state_dict(torch.load(best_name))
    

def test_max_auc(model, test_loader, device):
    '''AUC based on max prediction of 4 views
    '''
    model.eval()
    y_pool, p_pool = [], []
    with torch.no_grad():
        for batch in test_loader:
            bat_X_1 = batch['images']['same-cc'].float().to(device)
            bat_X_2 = batch['images']['same-mlo'].float().to(device)
            bat_X_3 = batch['images']['opposite-cc'].float().to(device)
            bat_X_4 = batch['images']['opposite-mlo'].float().to(device)
            bat_y = batch['label'].long().to(device)
            #bat_logit_1, _, _ = model(bat_X_1)
            #bat_logit_2, _, _ = model(bat_X_2)
            #bat_logit_3, _, _ = model(bat_X_3)
            #bat_logit_4, _, _ = model(bat_X_4)
            
            bat_logit_1 = model(bat_X_1)[:, :2, 1]
            bat_logit_2 = model(bat_X_2)[:, :2, 1]
            bat_logit_3 = model(bat_X_3)[:, :2, 1]
            bat_logit_4 = model(bat_X_4)[:, :2, 1]
            
            bat_p_1 = F.softmax(bat_logit_1)[:, 1]
            bat_p_2 = F.softmax(bat_logit_2)[:, 1]
            bat_p_3 = F.softmax(bat_logit_3)[:, 1]
            bat_p_4 = F.softmax(bat_logit_4)[:, 1]
            bat_p = torch.stack(
                [bat_p_1, bat_p_2, bat_p_3, bat_p_4], 
                dim=1)
            p_pool.append(bat_p)
            y_pool.append(bat_y)
        ys = torch.cat(y_pool)
        ps = torch.cat(p_pool)
        max_ps = ps.max(1).values
        ys = ys.cpu().detach().numpy()
        max_ps = max_ps.cpu().detach().numpy()
        test_auc = roc_auc_score(ys, max_ps)
        return test_auc

def val_loss(model, val_loader, device, return_auc=False):
    '''Avg loss and AUC for an entire data loader
    '''
    criterion_ = nn.NLLLoss(reduction='sum').to(device)
    model.eval()
    val_loss, n = 0, 0
    y_pool, p_pool = [], []
    with torch.no_grad():
        for batch in val_loader:
            bat_X_1 = batch['images']['same-cc'].float().to(device)
            bat_X_2 = batch['images']['same-mlo'].float().to(device)
            
            #print(bat_X_1.shape)
            #ii = cv2.imread(bat_X_1)
            #print(ii.shape)
            #bat_X_1 = torch.tensor(rgb2gray(bat_X_1.cpu().detach().numpy().reshape(2048, 1664, 3)).reshape(-1, 1, 2048, 1664)).float().to(device)
            #bat_X_2 = torch.tensor(rgb2gray(bat_X_2.cpu().detach().numpy().reshape(2048, 1664, 3)).reshape(-1, 1, 2048, 1664)).float().to(device)
            #print(bat_X_1.shape)
            
            
            bat_y = batch['label'].long().to(device)
            bat_logit_1 = model(bat_X_1)[:, :2, 1]
            bat_logit_2 = model(bat_X_2)[:, :2, 1]
            
            n += len(bat_X_1)
            n += len(bat_X_2)
            
            
            bat_logp_1 = F.log_softmax(bat_logit_1)
            bat_logp_2 = F.log_softmax(bat_logit_2)
            loss_1 = criterion_(bat_logp_1, bat_y)
            loss_2 = criterion_(bat_logp_2, bat_y)
            
            val_loss += loss_1.item()
            val_loss += loss_2.item()

            
            y_pool.append(bat_y)
            y_pool.append(bat_y)  # same label, different view.
            p_pool.append(torch.exp(bat_logp_1))
            p_pool.append(torch.exp(bat_logp_2))
        
            
        if return_auc:
            ys = torch.cat(y_pool)
            ps = torch.cat(p_pool)
            ys = ys.cpu().detach().numpy()
            ps = ps.cpu().detach().numpy()
            val_auc = roc_auc_score(ys, ps[:, 1])
            
            auc = val_loss/n, val_auc
            
            return auc
    return val_loss/n