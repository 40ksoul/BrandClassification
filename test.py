import os
import sys
import torchvision.models as models
import torch
import brand
import densenet
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import pandas as pd
import numpy as np
import pretrainedmodels

_NUM_CLASSES = 100

_BATCH_SIZE = 8

_TEST_AUGTIMES = 5

_MODELS_PATH = 'models/senet154/'

_EPOCH = '22'

_NETWORK_TYPE = 'senet154'

def main():
  rootpath = 'data/test/'
  txtpath = 'data/test.txt'
  inp_pre = 556
  inp_res= 512

  print('Start Testing!')
  test_dataset = brand.Brand(txtpath,rootpath,num_classes=_NUM_CLASSES, inp_pre=inp_pre, inp_res=inp_res, train=False, val=False)
  test_loader = DataLoader(dataset=test_dataset,batch_size=_BATCH_SIZE,shuffle=False)
    
  if _NETWORK_TYPE == 'resnet-v2':
    Network = resnet.resnet50(pretrained=True,v2=True,dp_ratio=0.5).cuda()
    Network.fc = torch.nn.Linear(2048,_ATTRIBUTE_INFO[attribute_type]).cuda()
  elif _NETWORK_TYPE == 'resnet':
    Network = resnet.resnet50(pretrained=True,v2=False,dp_ratio=0.5).cuda()
    Network.fc = torch.nn.Linear(2048,_ATTRIBUTE_INFO[attribute_type]).cuda()
  elif _NETWORK_TYPE == 'senet':
    Network = pretrainedmodels.__dict__['se_resnet50'](pretrained=False).cuda()
    Network.avg_pool = torch.nn.AvgPool2d(16, stride=1)
    Network.last_linear = torch.nn.Linear(2048,_NUM_CLASSES).cuda()
  elif _NETWORK_TYPE == 'senet154':
    Network = pretrainedmodels.__dict__['senet154']().cuda()
    Network.avg_pool = torch.nn.AvgPool2d(16, stride=1)
    Network.last_linear = torch.nn.Linear(2048,_NUM_CLASSES).cuda()
  elif _NETWORK_TYPE == 'densenet':
    Network = densenet.densenet121(pretrained=False,num_classes=_NUM_CLASSES).cuda()
  else:
    raise 'Unknown network type error!'
  Network.eval()
  # print(resnet.state_dict().keys())
  para_dict = torch.load(_MODELS_PATH+'/'+str(_EPOCH)+'-params.pk1')
  # print('para_dict')
  # print(para_dict.keys())
  Network.load_state_dict(para_dict)
    
  attribute_prob = []
  attribute_mark = []
  for test_num in range(_TEST_AUGTIMES):
    test_count = 0
    print('===> aug-testing '+str(test_num+1))
    for batch_x, batch_y,meta in tqdm(test_loader):
      batch_x = Variable(batch_x,volatile=True).cuda()
      out = Network(batch_x)
      out = torch.nn.functional.softmax(out,dim=1)
        
      out = out.cpu().data.numpy()
        
      for i in range(len(out)):
        if test_count >= len(attribute_prob):
          attribute_prob.append([])
          attribute_mark.append(meta['img_path'][i])
        attribute_prob[test_count].append(out[i])
        test_count += 1
  
  results = []
  for i in range(len(attribute_prob)):
    mean_prob = np.sum(attribute_prob[i],axis=0)/_TEST_AUGTIMES
    resultstr = attribute_mark[i][len(rootpath):len(attribute_mark[i])] + ' ' + str(np.argmax(mean_prob)+1)
    results.append(resultstr)
  print('done!')
  # print(meta)
  with open('results/result.txt','w') as f:
    for i in range(len(results)):
      f.write(results[i]+'\n')


def getAnnoList(rootpath,csv_path):
  csv_file = pd.read_csv(rootpath + csv_path,header=None).values
  anno = {}
  for row in tqdm(csv_file):
    if row[1] not in anno.keys():
      anno[row[1]] = []
    single = {}
    single['path'] = rootpath + row[0]
    anno[row[1]].append(single)
  return anno

def trainRecord():
  csv_path = 'data/val.csv'

  csv_file = pd.read_csv(csv_path,header=None).values
  count = 0
  anno = {}
  outlist = []
  print('Data Preprocessing!')
  for row in tqdm(csv_file):
    singlelist = []
    for unit in row:
      singlelist.append(unit)
    anno[row[0]]=count
    count += 1
    outlist.append(singlelist)
  print('Start Testing!')


  test_dataset = attribute.Attribute(csv_path,True,attribute_type,train=False,is_full_mode=True)
  test_loader = DataLoader(dataset=test_dataset,batch_size=_BATCH_SIZE,shuffle=False)

  if _NETWORK_TYPE == 'resnet-v2':
    Network = resnet.resnet50(pretrained=True,v2=True,dp_ratio=0.5).cuda()
    Network.fc = torch.nn.Linear(2048,_ATTRIBUTE_INFO[attribute_type]).cuda()
  elif _NETWORK_TYPE == 'resnet':
    Network = resnet.resnet50(pretrained=True,v2=False,dp_ratio=0.5).cuda()
    Network.fc = torch.nn.Linear(2048,_ATTRIBUTE_INFO[attribute_type]).cuda()
  elif _NETWORK_TYPE == 'densenet':
    Network = densenet.densenet121(pretrained=True,num_classes=_ATTRIBUTE_INFO[attribute_type],dp_ratio=0.5).cuda()
  else:
    raise 'Unknown network type error!'
  Network.eval()
  # print(resnet.state_dict().keys())
  para_dict = torch.load(_ATTRIBUTE_MODELS[attribute_type]+attribute_type+'/'+str(_ATTRIBUTE_EPOCH[attribute_type])+'-params.pk1')
  # print('para_dict')
  # print(para_dict.keys())
  Network.load_state_dict(para_dict)
    
  attribute_prob = []
  attribute_mark = []
  for test_num in range(_TEST_AUGTIMES):
    test_count = 0
    print('===>'+attribute_type+' aug-testing '+str(test_num+1))
    for batch_x, batch_y,meta in tqdm(test_loader):
      batch_x = Variable(batch_x,volatile=True).cuda()
      out = Network(batch_x)
      out = torch.nn.functional.softmax(out,dim=1)
      
      out = out.cpu().data.numpy()
        
      for i in range(len(out)):
        if test_count >= len(attribute_prob):
          attribute_prob.append([])
          attribute_mark.append(meta['img_path'][i])
        attribute_prob[test_count].append(out[i])
        test_count += 1

  for i in range(len(attribute_prob)):
    mean_prob = np.sum(attribute_prob[i],axis=0)/_TEST_AUGTIMES
    resultstr = ''
    for j in range(len(mean_prob)):
      resultstr += '{:.4f};'.format(mean_prob[j])
    resultstr = resultstr[0:len(resultstr)-1]
    outlist[anno[attribute_mark[i]]].append(resultstr)
    # print(meta)
  print(attribute_type+' done!')
  out_csv = pd.DataFrame(np.array(outlist))
  out_csv.to_csv('full_label_logistic_val2.csv',index=False,header=None)

def testRecord():
  rootpath = 'data/Test/'
  csv_path = 'Tests/question.csv'

  csv_file = pd.read_csv(rootpath+csv_path,header=None).values
  count = 0
  anno = {}
  outlist = []
  print('Data Preprocessing!')
  for row in tqdm(csv_file):
    singlelist = []
    for unit in row:
      singlelist.append(unit)
    anno[row[0]]=count
    count += 1
    outlist.append(singlelist)
  print('Start Testing!')

  for attribute_type in _ATTRIBUTE_INFO.keys():
    test_dataset = attribute.Attribute(csv_path,True,attribute_type,train=False,is_full_mode=True,csv_prefix=rootpath)
    test_loader = DataLoader(dataset=test_dataset,batch_size=_BATCH_SIZE,shuffle=False)

    if _NETWORK_TYPE == 'resnet-v2':
      Network = resnet.resnet50(pretrained=True,v2=True,dp_ratio=0.5).cuda()
      Network.fc = torch.nn.Linear(2048,_ATTRIBUTE_INFO[attribute_type]).cuda()
    elif _NETWORK_TYPE == 'resnet':
      Network = resnet.resnet50(pretrained=True,v2=False,dp_ratio=0.5).cuda()
      Network.fc = torch.nn.Linear(2048,_ATTRIBUTE_INFO[attribute_type]).cuda()
    elif _NETWORK_TYPE == 'densenet':
      Network = densenet.densenet121(pretrained=True,num_classes=_ATTRIBUTE_INFO[attribute_type],dp_ratio=0.5).cuda()
    else:
      raise 'Unknown network type error!'
    Network.eval()
    # print(resnet.state_dict().keys())
    para_dict = torch.load(_ATTRIBUTE_MODELS[attribute_type]+attribute_type+'/'+str(_ATTRIBUTE_EPOCH[attribute_type])+'-params.pk1')
    # print('para_dict')
    # print(para_dict.keys())
    Network.load_state_dict(para_dict)
    
    attribute_prob = []
    attribute_mark = []
    for test_num in range(_TEST_AUGTIMES):
      test_count = 0
      print('===>'+attribute_type+' aug-testing '+str(test_num+1))
      for batch_x, batch_y,meta in tqdm(test_loader):
        batch_x = Variable(batch_x,volatile=True).cuda()
        out = Network(batch_x)
        out = torch.nn.functional.softmax(out,dim=1)
        
        out = out.cpu().data.numpy()
        
        for i in range(len(out)):
          if test_count >= len(attribute_prob):
            attribute_prob.append([])
            attribute_mark.append(meta['img_path'][i])
          attribute_prob[test_count].append(out[i])
          test_count += 1

    for i in range(len(attribute_prob)):
      mean_prob = np.sum(attribute_prob[i],axis=0)/_TEST_AUGTIMES
      resultstr = ''
      for j in range(len(mean_prob)):
        resultstr += '{:.4f};'.format(mean_prob[j])
      resultstr = resultstr[0:len(resultstr)-1]
      outlist[anno[attribute_mark[i][len(rootpath):len(attribute_mark[i])]]].append(resultstr)
      # print(meta)
    print(attribute_type+' done!')
  out_csv = pd.DataFrame(np.array(outlist))
  out_csv.to_csv('answer_full_label.csv',index=False,header=None)

if __name__ == '__main__':
  main()
  #trainRecord()
  #testRecord()
