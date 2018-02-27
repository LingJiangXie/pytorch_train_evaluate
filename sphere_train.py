import os
import torch
import torch.nn as nn
from torch.nn import Parameter
import math

import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as dset
from torch.utils.data import DataLoader,Dataset

from PIL import Image

from torch import optim
import numpy as np

import os
import torchvision.transforms.functional as ttf

def myphi(x,m):
    x = x * m
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)

class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(ww) # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_theta.data.acos())
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta,self.m)
            phi_theta = phi_theta.clamp(-1*self.m,1)

        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)
        output = (cos_theta,phi_theta)
        return output # size=(B,Classnum,2)


class AngleLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma   = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte()
        index = Variable(index)

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output,dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss


class sphere20a(nn.Module):
    def __init__(self,classnum=10575,feature=False):
        super(sphere20a, self).__init__()
        self.classnum = classnum
        self.feature = feature
        #input = B*3*112*96
        self.conv1_1 = nn.Conv2d(3,64,3,2,1) #=>B*64*56*48
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64,64,3,1,1)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64,64,3,1,1)
        self.relu1_3 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64,128,3,2,1) #=>B*128*28*24
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128,128,3,1,1)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128,128,3,1,1)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128,128,3,1,1) #=>B*128*28*24
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128,128,3,1,1)
        self.relu2_5 = nn.PReLU(128)


        self.conv3_1 = nn.Conv2d(128,256,3,2,1) #=>B*256*14*12
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256,256,3,1,1)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256,256,3,1,1)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256,256,3,1,1)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256,256,3,1,1)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256,256,3,1,1)
        self.relu3_9 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256,512,3,2,1) #=>B*512*7*6
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512,512,3,1,1)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512,512,3,1,1)
        self.relu4_3 = nn.PReLU(512)

        self.fc5 = nn.Linear(512*7*6,512)
        #self.bn = nn.BatchNorm2d(512)
        self.fc6 = AngleLinear(512,self.classnum)


    def forward(self, x):
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))

        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))

        x = x.view(x.size(0),-1)
        x = self.fc5(x)
        #x = self.bn(x)
        if self.feature: return x

        x = self.fc6(x)
        return x


transform_train = transforms.Compose([
    #transforms.Resize(127),
    transforms.RandomCrop((112,96)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),

])


train_dataset = dset.ImageFolder('/home/dany/Documents/datasets/CASIA-maxpy-clean_116_100', transform=transform_train)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=128,
                                           shuffle=True,num_workers=10)

net=sphere20a()
net.load_state_dict(torch.load('/home/dany/Downloads/sphereface_pytorch-master/net5c.pkl'))
net=net.cuda()
# Loss and Optimizer
criterion =AngleLoss()

lr = 0.001

#optimizer = torch.optim.Adam(net.parameters(), lr=lr)
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

# Training
for epoch in range(1):

    train_loss = 0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs[0].data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()

        if (i + 1) % 100 == 0:
            print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f | Acc: %.4f%% (%d/%d)" % (
            epoch + 1, 100, i + 1, 500, train_loss / (i + 1), 100. * correct / total, correct, total))



# Save the Model
torch.save(net.state_dict(), 'net6c.pkl')


def KFold(n=6000, n_folds=10, shuffle=False):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[int(i*n/n_folds):int((i+1)*n/n_folds)]
        train = list(set(base)-set(test))
        folds.append([train,test])
    return folds

def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0*np.count_nonzero(y_true==y_predict)/len(y_true)
    return accuracy

def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold



test_dir_str='/home/dany/Documents/datasets/lfw_112_96/'

with open('/home/dany/Downloads/sphereface_pytorch-master/data/pairs.txt') as f:
    pairs_lines = f.readlines()[1:]


predicts=[]
net=sphere20a()
net.load_state_dict(torch.load('/home/dany/Downloads/sphereface_pytorch-master/net6c.pkl'))
net.cuda()
net.eval()
net.feature = True


for i in range(6000):
    p = pairs_lines[i].replace('\n','').split('\t')

    if 3==len(p):
        sameflag = 1
        name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
        name2 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[2]))
        name1m = p[0] + '/' + 'a'+p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
        name2m = p[0] + '/' + 'a'+p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
    if 4==len(p):
        sameflag = 0
        name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
        name2 = p[2]+'/'+p[2]+'_'+'{:04}.jpg'.format(int(p[3]))
        name1m = p[0] + '/' + 'a'+p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
        name2m = p[2] + '/' + 'a'+p[2] + '_' + '{:04}.jpg'.format(int(p[3]))

    img1_path = os.path.join(test_dir_str, name1)
    img2_path = os.path.join(test_dir_str, name2)
    img1m_path = os.path.join(test_dir_str, name1m)
    img2m_path = os.path.join(test_dir_str, name2m)

    '''
    img1 = ttf.to_tensor(ttf.resize(Image.open(img1_path), 112))
    img2 = ttf.to_tensor(ttf.resize(Image.open(img2_path), 112))
    img1m = ttf.to_tensor(ttf.resize(Image.open(img1m_path), 112))
    img2m = ttf.to_tensor(ttf.resize(Image.open(img2m_path), 112))
    '''
    img1 = ttf.to_tensor(Image.open(img1_path))
    img2 = ttf.to_tensor(Image.open(img2_path))
    img1m = ttf.to_tensor(Image.open(img1m_path))
    img2m = ttf.to_tensor(Image.open(img2m_path))

    img1 = ttf.normalize(img1, [0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
    img2 = ttf.normalize(img2, [0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
    img1m = ttf.normalize(img1m, [0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
    img2m = ttf.normalize(img2m, [0.5, 0.5, 0.5], [0.2, 0.2, 0.2])

    img1=Variable(img1.cuda())
    img2 = Variable(img2.cuda())
    img1m = Variable(img1m.cuda())
    img2m = Variable(img2m.cuda())

    imgs=torch.stack([img1,img2,img1m,img2m], dim=0)

    #print(imgs) torch.cuda.FloatTensor of size 2x3x160x160 (GPU 0)

    output = net(imgs)
    f = output.data
    f1, f2 ,f1m, f2m= f[0], f[1],f[2], f[3]

    #f1, f2 = f[0], f[1]

    f1 = torch.cat((f1,f1m), 0)
    f2 = torch.cat((f2,f2m), 0)

    cosdistance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
    predicts.append('{}\t{}\t{}\t{}\n'.format(name1, name2, cosdistance, sameflag))


accuracy = []
thd = []
folds = KFold(n=6000, n_folds=10, shuffle=False)
thresholds = np.arange(-1.0, 1.0, 0.005)
predicts = np.array(list(map(lambda line:line.strip('\n').split(), predicts)))

for idx, (train, test) in enumerate(folds):
    best_thresh = find_best_threshold(thresholds, predicts[train])
    accuracy.append(eval_acc(best_thresh, predicts[test]))
    thd.append(best_thresh)
print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))



#test roc
'''
def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold

def eval_tar_far(threshold, diff):
    y_true = []

    y_predict = []

    for d in diff:

        if float(d[2]) > threshold:
            y_predict.append(1)
        else:
            y_predict.append(0)

        y_true.append(int(d[3]))

    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    tp = np.sum(np.logical_and(y_predict, y_true))
    fp = np.sum(np.logical_and(y_predict, np.logical_not(y_true)))
    n_same = np.sum(y_true)
    fn = n_same - tp
    n_diff = np.sum(np.logical_not(y_true))
    tn = n_diff - fp

    tar = float(tp) / float(tp + fn)
    far = float(fp) / float(fp + tn)

    return tar, far

def my_calculate_val(thresholds, nrof_pairs, predicts, far_target, nrof_folds=10):

    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):

            _, far_train[threshold_idx] = eval_tar_far(threshold, predicts[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = eval_tar_far(threshold, predicts[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean

preds=np.load('/home/dany/Downloads/sphereface_pytorch-master/predicts.npy')
thresholds = np.arange(-1.0, 1.0, 0.005)
nrof_folds=10
k_fold = KFold(n_splits=nrof_folds, shuffle=True)
indices = np.arange(6000)

print(my_calculate_val(thresholds, 6000, preds, 1e-2, nrof_folds=10))
'''
'''

def eval_acc(threshold, diff):

    y_true = []

    y_predict = []


    for d in diff:

        if float(d[2]) > threshold:
            y_predict.append(1)
        else:
            y_predict.append(0)


        y_true.append(int(d[3]))

    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    tp = np.sum(np.logical_and(y_predict, y_true))
    fp = np.sum(np.logical_and(y_predict, np.logical_not(y_true)))
    n_same = np.sum(y_true)
    fn=n_same-tp
    n_diff = np.sum(np.logical_not(y_true))
    tn=n_diff-fp

    accuracy = float(tp+tn)/float(tp+fn+fp+tn)

    return accuracy

'''