
import torchvision.datasets as dset
from torch.utils.data import DataLoader,Dataset
import numpy as np
import torchvision.transforms.functional as ttf
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import  transforms
from torch.autograd import Variable


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, loss_weight=1.0):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.loss_weight = loss_weight
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

        self.use_cuda = False

    def forward(self, y, feat):
        # torch.histc can only be implemented on CPU
    	# To calculate the total number of every class in one mini-batch. See Equation 4 in the paper
        if self.use_cuda:
            hist = Variable(torch.histc(y.cpu().data.float(),bins=self.num_classes,min=0,max=self.num_classes) + 1).cuda()
        else:
            hist = Variable(torch.histc(y.data.float(),bins=self.num_classes,min=0,max=self.num_classes) + 1)

        centers_count = hist.index_select(0,y.long())


        # To squeeze the Tenosr
        batch_size = feat.size()[0]
        feat = feat.view(batch_size, 1, 1, -1).squeeze()
        # To check the dim of centers and features
        if feat.size()[1] != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim,feat.size()[1]))

        centers_pred = self.centers.index_select(0, y.long())

        centers_pred_norm = centers_pred.renorm(2, 1, 1e-5).mul(1e5)

        feat_norm = feat.renorm(2, 1, 1e-5).mul(1e5)

        diff = feat_norm - centers_pred_norm

        loss = self.loss_weight * 1 / 2.0 * (diff.pow(2).sum(1) / centers_count).sum()

        return loss

    def cuda(self, device_id=None):

        self.use_cuda = True
        return self._apply(lambda t: t.cuda(device_id))


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

        self.fc5 = nn.Linear(512*7*7,512)
        self.fc6 = nn.Linear(512,self.classnum)


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
        ip1 = self.fc5(x)
        if self.feature: return ip1
        ip2 = self.fc6(ip1)
        return ip1, F.log_softmax(ip2,dim=1)


transform_train = transforms.Compose([
    transforms.Resize(116, interpolation=2),
    transforms.RandomCrop(112),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

])


train_dataset = dset.ImageFolder('/home/dany/Documents/datasets/CASIA-maxpy-clean_mtcnnpy_182', transform=transform_train)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=256,
                                           shuffle=True,num_workers=16)

use_cuda = True

net=sphere20a()
net.load_state_dict(torch.load('/home/dany/Downloads/sphereface_pytorch-master/net16.pkl'))
nllloss = nn.NLLLoss()  # CrossEntropyLoss = log_softmax + NLLLoss

loss_weight = 1.0
centerloss = CenterLoss(10575, 512, loss_weight)
centerloss.load_state_dict(torch.load('/home/dany/Downloads/sphereface_pytorch-master/centerloss16.pkl'))

if use_cuda:

    nllloss = nllloss.cuda()
    centerloss = centerloss.cuda()
    net = net.cuda()

criterion = [nllloss, centerloss]

optimizer1 = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)

optimizer2 = optim.SGD(centerloss.parameters(), lr=0.0001)

# Training
for epoch in range(4):

    train_loss = 0
    s_loss=0
    c_loss=0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())

        # Forward + Backward + Optimize
        optimizer1.zero_grad()
        optimizer2.zero_grad()

        ip1, pred = net(images)

        loss = criterion[0](pred, labels) + criterion[1](labels, ip1)


        loss.backward()

        optimizer1.step()
        optimizer2.step()

        train_loss += loss.data[0]

        s_loss += criterion[0](pred, labels).data[0]

        c_loss += criterion[1](labels, ip1).data[0]

        _, predicted = torch.max(pred.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()

        if (i + 1) % 100 == 0:
            print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f S-Loss: %.4f C-Loss: %.4f | Acc: %.4f%% (%d/%d)" % (
            epoch + 1, 100, i + 1, 500, train_loss / (i + 1),s_loss / (i + 1),c_loss / (i + 1), 100. * correct / total, correct, total))


# Save the Model
torch.save(net.state_dict(), 'net20.pkl')
torch.save(centerloss.state_dict(), 'centerloss20.pkl')

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



test_dir_str='/home/dany/Documents/datasets/lfw_mtcnn_160/'

with open('/home/dany/Downloads/sphereface_pytorch-master/data/pairs.txt') as f:
    pairs_lines = f.readlines()[1:]


predicts=[]
net=sphere20a()
net.load_state_dict(torch.load('/home/dany/Downloads/sphereface_pytorch-master/net20.pkl'))
net.cuda()
net.eval()
net.feature = True


for i in range(6000):
    p = pairs_lines[i].replace('\n','').split('\t')

    if 3==len(p):
        sameflag = 1
        name1 = p[0]+'/'+p[0]+'_'+'{:04}.png'.format(int(p[1]))
        name2 = p[0]+'/'+p[0]+'_'+'{:04}.png'.format(int(p[2]))
        name1m = p[0] + '/' + 'mx'+p[0] + '_' + '{:04}.png'.format(int(p[1]))
        name2m = p[0] + '/' + 'mx'+p[0] + '_' + '{:04}.png'.format(int(p[2]))
    if 4==len(p):
        sameflag = 0
        name1 = p[0]+'/'+p[0]+'_'+'{:04}.png'.format(int(p[1]))
        name2 = p[2]+'/'+p[2]+'_'+'{:04}.png'.format(int(p[3]))
        name1m = p[0] + '/' + 'mx'+p[0] + '_' + '{:04}.png'.format(int(p[1]))
        name2m = p[2] + '/' + 'mx'+p[2] + '_' + '{:04}.png'.format(int(p[3]))

    img1_path = os.path.join(test_dir_str, name1)
    img2_path = os.path.join(test_dir_str, name2)
    img1m_path = os.path.join(test_dir_str, name1m)
    img2m_path = os.path.join(test_dir_str, name2m)


    img1 = ttf.to_tensor(ttf.resize(Image.open(img1_path), 112, 2))
    img2 = ttf.to_tensor(ttf.resize(Image.open(img2_path), 112, 2))
    img1m = ttf.to_tensor(ttf.resize(Image.open(img1m_path), 112, 2))
    img2m = ttf.to_tensor(ttf.resize(Image.open(img2m_path), 112, 2))

    img1 = ttf.normalize(img1, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img2 = ttf.normalize(img2, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img1m = ttf.normalize(img1m, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img2m = ttf.normalize(img2m, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    img1=Variable(img1.cuda())
    img2 = Variable(img2.cuda())
    img1m = Variable(img1m.cuda())
    img2m = Variable(img2m.cuda())

    imgs=torch.stack([img1,img2,img1m,img2m], dim=0)

    #print(imgs) torch.cuda.FloatTensor of size 2x3x160x160 (GPU 0)

    output = net(imgs)
    f = output.data
    f1, f2 ,f1m, f2m= f[0], f[1],f[2], f[3]

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