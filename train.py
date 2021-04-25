import model
import reader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os
import copy
import yaml
import torch.backends.cudnn as cudnn

if __name__ == "__main__":
  config = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)
  config = config["train"]
  cudnn.benchmark=True

  imagepath = config["data"]["image"]
  labelpath = config["data"]["label"]
  modelname = config["save"]["model_name"]

  folder = os.listdir(labelpath)
  folder.sort()

  # i represents the i-th folder used as the test set.
  i = int(sys.argv[2])

  if i in list(range(15)):
    trains = copy.deepcopy(folder)
    tests = trains.pop(i)
    print(f"Train Set:{trains}")
    print(f"Test Set:{tests}")

    trainlabelpath = [os.path.join(labelpath, j) for j in trains] 

    savepath = os.path.join(config["save"]["save_path"], f"checkpoint/{tests}")
    if not os.path.exists(savepath):
      os.makedirs(savepath)
  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("Read data")
    dataset = reader.txtload(trainlabelpath, imagepath, config["params"]["batch_size"], shuffle=True, num_workers=6, header=True)

    print("Model building")
    net = model.model()
    net.train()
    net.to(device)

    print("optimizer building")
    lossfunc = config["params"]["loss"]
    loss_op = getattr(nn, lossfunc)().cuda()
    base_lr = config["params"]["lr"]

    decaysteps = config["params"]["decay_step"]
    decayratio = config["params"]["decay"]

    optimizer = optim.Adam(net.parameters(),lr=base_lr, betas=(0.9,0.95))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decaysteps, gamma=decayratio)

    print("Traning")
    length = len(dataset)
    total = length * config["params"]["epoch"]
    cur = 0
    timebegin = time.time()
    with open(os.path.join(savepath, "train_log"), 'w') as outfile:
      for epoch in range(1, config["params"]["epoch"]+1):
        for i, (data, label) in enumerate(dataset):

          # Acquire data
          data["eye"] = data["eye"].to(device)
          data['head_pose'] = data['head_pose'].to(device)
          label = label.to(device)
   
          # forward
          gaze = net(data)

          # loss calculation
          loss = loss_op(gaze, label)
          optimizer.zero_grad()

          # backward
          loss.backward()
          optimizer.step()
          scheduler.step()
          cur += 1

          # print logs
          if i % 20 == 0:
            timeend = time.time()
            resttime = (timeend - timebegin)/cur * (total-cur)/3600
            log = f"[{epoch}/{config['params']['epoch']}]: [{i}/{length}] loss:{loss} lr:{base_lr}, rest time:{resttime:.2f}h"
            print(log)
            outfile.write(log + "\n")
            sys.stdout.flush()   
            outfile.flush()

        if epoch % config["save"]["step"] == 0:
          torch.save(net.state_dict(), os.path.join(savepath, f"Iter_{epoch}_{modelname}.pt"))

