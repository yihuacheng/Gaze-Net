import numpy as np
import cv2 
import os
from torch.utils.data import Dataset, DataLoader
import torch

def gazeto2d(gaze):
  yaw = np.arctan2(-gaze[0], -gaze[2])
  pitch = np.arcsin(-gaze[1])
  return np.array([yaw, pitch])

class loader(Dataset): 
  def __init__(self, path, root, header=True):
    self.lines = []
    if isinstance(path, list):
      for i in path:
        with open(i) as f:
          line = f.readlines()
          if header: line.pop(0)
          self.lines.extend(line)
    else:
      with open(path) as f:
        self.lines = f.readlines()
        if header: self.lines.pop(0)

    self.root = root

  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    line = self.lines[idx]
    line = line.strip().split(" ")

    name = line[1]
    gaze2d = line[5]
    head2d = line[6]
    eye = line[0]

    label = np.array(gaze2d.split(",")).astype("float")
    label = torch.from_numpy(label).type(torch.FloatTensor)

    headpose = np.array(head2d.split(",")).astype("float")
    headpose = torch.from_numpy(headpose).type(torch.FloatTensor)

    img = cv2.imread(os.path.join(self.root, eye))/255.0
    img = img.transpose(2, 0, 1)

    info = {"eye":torch.from_numpy(img).type(torch.FloatTensor),
            "head_pose":headpose,
            "name":name}

    return info, label

def txtload(labelpath, imagepath, batch_size, shuffle=True, num_workers=0, header=True):
  dataset = loader(labelpath, imagepath, header)
  print(f"[Read Data]: Total num: {len(dataset)}")
  print(f"[Read Data]: Label path: {labelpath}")
  load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
  return load


if __name__ == "__main__":
  path = './p00.label'
  d = loader(path)
  print(len(d))
  (data, label) = d.__getitem__(0)

