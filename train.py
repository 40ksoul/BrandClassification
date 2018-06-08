import os
import sys
import densenet
import torch
import brand
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
import pretrainedmodels


_MAX_EPOCH = 50

_BATCH_SIZE = 5

_MODELS_PATH = 'models/senet154'

_NETWORK_TYPE = 'senet154'

_NUM_CLASSES = 100

_IS_VAL = False

def main():
  inp_pre = 556
  inp_res= 512

  if _IS_VAL:
    train_dataset = brand.Brand('data/trainb.txt','data/train/',_NUM_CLASSES,inp_pre=inp_pre, inp_res=inp_res, train=True, val=False)
    train_loader = DataLoader(dataset=train_dataset,batch_size=_BATCH_SIZE,shuffle=True)
    val_dataset =  brand.Brand('data/val.txt','data/train/',_NUM_CLASSES,inp_pre=inp_pre, inp_res=inp_res, train=False, val=True)
    val_loader = DataLoader(dataset=val_dataset,batch_size=_BATCH_SIZE,shuffle=False)
  else:
    train_dataset = brand.Brand('data/train.txt','data/train/',_NUM_CLASSES,inp_pre=inp_pre, inp_res=inp_res, train=True, val=False)
    train_loader = DataLoader(dataset=train_dataset,batch_size=_BATCH_SIZE,shuffle=True)

  if _NETWORK_TYPE == 'senet':
    Network = pretrainedmodels.__dict__['se_resnet50']().cuda()
    Network.avg_pool = torch.nn.AvgPool2d(16, stride=1)
    Network.last_linear = torch.nn.Linear(2048,_NUM_CLASSES).cuda()
  elif _NETWORK_TYPE == 'senet154':
    Network = pretrainedmodels.__dict__['senet154']().cuda()
    Network.avg_pool = torch.nn.AvgPool2d(16, stride=1)
    Network.last_linear = torch.nn.Linear(2048,_NUM_CLASSES).cuda()
  else:
    raise 'Unknown network type error!'

  optimizer = torch.optim.Adam(Network.parameters(),lr=1e-4,weight_decay=0.01)
  schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer,[5,10,20,30,40,50],0.333)
  # if _IS_APNORM:
  #   loss_func = APnorm_loss.APnormLoss(class_num=_NUM_LABELS,beta=0.1).cuda()
  # else:
  loss_func = torch.nn.CrossEntropyLoss().cuda()
  # loss_func = torch.nn.BCEWithLogitsLoss().cuda()

  if not os.path.isdir(_MODELS_PATH):
    os.mkdir(_MODELS_PATH)

  log_file = open(_MODELS_PATH+'/train.log','a')
  log_file.write(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+'|| Training begin! \n')
  log_file.write('epoch | Train Loss | Train Acc | Val Loss | Val Acc \n')
  log_file.close()


  for epoch in range(_MAX_EPOCH):
    schedular.step()
    print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    Network.train()
    train_loss = 0.
    train_acc = 0.
    count = 0
    for batch_x, batch_y,meta in train_loader:
      count += 1
      if(count%100==0):
        print('Iteration {:d} done! Train Loss: {:.6f}, Acc: {:.6f}'.format(count,train_loss / (count * _BATCH_SIZE),train_acc / (count * _BATCH_SIZE)))
      batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda()

      batch_y = torch.max(batch_y,1)[1]
      out = Network(batch_x)
      loss = loss_func(out,batch_y)
      pred = torch.max(out,1)[1]

      train_loss += loss.data[0]  
      train_correct = (pred == batch_y).sum()
      train_acc +=train_correct.data[0]
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(
      train_loss / (len(train_dataset)), train_acc / len(train_dataset)
      ))
    # print('Example Training Output:' + str(out))
    if(epoch+1>=10 and epoch%2==0 or epoch==49):
      torch.save(Network.state_dict(),_MODELS_PATH+'/'+str(epoch)+'-params.pk1')
    if _IS_VAL:
      # evaluation-----------------------------
      print('evaluating========>')
      Network.eval()
      eval_loss = 0.
      eval_acc = 0.
      for batch_x,batch_y,meta in val_loader:
        batch_x, batch_y = Variable(batch_x,volatile=True).cuda(), Variable(batch_y,volatile=True).cuda()

        batch_y = torch.max(batch_y,1)[1]
        out = Network(batch_x)
        loss = loss_func(out,batch_y)
        pred = torch.max(out,1)[1]

        eval_loss += loss.data[0]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.data[0]
      print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
        eval_loss / (len(val_dataset)), eval_acc / len(val_dataset)
        ))
    log_file = open(_MODELS_PATH+'/train.log','a')
    if _IS_VAL:
      log_file.write('{:3d} {:.6f} {:.6f} {:.6f} {:.6f} \n'.format(epoch,
        train_loss / (len(train_dataset)), train_acc / len(train_dataset), 
        eval_loss / (len(val_dataset)), eval_acc / len(val_dataset)))
    else:
      log_file.write('{:3d} {:.6f} {:.6f}\n'.format(epoch,
        train_loss / len(train_dataset), train_acc / len(train_dataset)))
    log_file.close()


if __name__ == '__main__':
  main()
