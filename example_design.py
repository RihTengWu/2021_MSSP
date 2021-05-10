import numpy as np
from torch.autograd import Variable
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import glob
import os

class Net(torch.nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		self.encoder = nn.Sequential(
			nn.Conv2d(in_channels=8,out_channels=256,kernel_size=3,padding=1,dilation=1,groups=1,bias=True), 
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=256,out_channels=64,kernel_size=3,padding=1,dilation=1,groups=1,bias=True),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2,stride=2,padding=0,dilation=1,return_indices=False,ceil_mode=False),
			nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,padding=1,dilation=1,groups=1,bias=True), 
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=32,out_channels=16,kernel_size=3,padding=1,dilation=1,groups=1,bias=True), 
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2,stride=2,padding=0,dilation=1,return_indices=False,ceil_mode=False),
			nn.Conv2d(in_channels=16,out_channels=8,kernel_size=3,padding=1,dilation=1,groups=1,bias=True), 
			nn.ReLU(inplace=True))

		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(in_channels=8,out_channels=16,kernel_size=3,padding=1,dilation=1,groups=1,bias=True), 
			nn.ReLU(inplace=True),
			nn.Upsample(size=[135,105], mode='bilinear',align_corners=True), # to ensure the final output size is identical to input size 
			nn.ConvTranspose2d(in_channels=16,out_channels=32,kernel_size=3,padding=1,dilation=1,groups=1,bias=True),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(in_channels=32,out_channels=64,kernel_size=3,padding=1,dilation=1,groups=1,bias=True),
			nn.ReLU(inplace=True),
			nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
			nn.ConvTranspose2d(in_channels=64,out_channels=256,kernel_size=3,padding=1,dilation=1,groups=1,bias=True),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(in_channels=256,out_channels=8,kernel_size=3,padding=1,dilation=1,groups=1,bias=True),
			nn.ReLU(inplace=True))

		# ==== geometry estimator
		self.c1 = nn.Sequential(
			nn.Linear(67*52*8, 18),
			nn.ReLU(inplace=True),
			nn.Linear(18, 15),
			nn.ReLU(inplace=True),
			nn.Linear(15, 13))
		
		self.c2 = nn.Sequential(
			nn.Linear(67*52*8, 18),
			nn.ReLU(inplace=True),
			nn.Linear(18, 15),
			nn.ReLU(inplace=True),
			nn.Linear(15, 13))
			
		self.c3 = nn.Sequential(
			nn.Linear(67*52*8, 18),
			nn.ReLU(inplace=True),
			nn.Linear(18, 15),
			nn.ReLU(inplace=True),
			nn.Linear(15, 13))
			
		self.c4 = nn.Sequential(
			nn.Linear(67*52*8, 18),
			nn.ReLU(inplace=True),
			nn.Linear(18, 15),
			nn.ReLU(inplace=True),
			nn.Linear(15, 13))
			

	def forward(self, x):
		x2 = self.encoder(x)
		# ==== autoencoder output
		x3 = self.decoder(x2)
		# ==== scatterer predictor output
		x4 = x2.view(x2.size(0), -1)
		y1 = self.c1(x4)
		y2 = self.c2(x4)
		y3 = self.c3(x4)
		y4 = self.c4(x4)
		return x3,y1,y2,y3,y4

# ==== load model
CASE = '62'
model = torch.load("model_3_300_"+CASE).cuda()
model.eval() # evaluation mode

# ==== load data
image_path = '/data/rihteng/ME/mehdi/20190918_4scatters/image_data_20200427_new_shapes_not_in_training/example_inputs.pt'
x = torch.load(image_path) # 5000, 200, 796, 12500Hz
num_test = x.shape[0]
result = np.zeros((num_test, 4))

# ==== scaling back 
MAX_r = 1.9075 # real part
MIN_r = -1.9146
MAX_i = 1.9392 # imaginary part
MIN_i = -1.8605

x[:,[0,2,4,6],:,:] = x[:,[0,2,4,6],:,:]*(MAX_r-MIN_r)+MIN_r # scale back to original pressure fields
x[:,[1,3,5,7],:,:] = x[:,[1,3,5,7],:,:]*(MAX_i-MIN_i)+MIN_i

# ==== scale separately
MAX_r = 1.8530 # real part
MIN_r = -1.8930
MAX_i = 1.8740 # imaginary part
MIN_i = -1.8822

x[:,0,:,:] = (x[:,0,:,:]-MIN_r)/(MAX_r-MIN_r) # 5000 Hz
x[:,1,:,:] = (x[:,1,:,:]-MIN_i)/(MAX_i-MIN_i)

MAX_r = 0.1181 # real part
MIN_r = -0.2540
MAX_i = 0.1495 # imaginary part
MIN_i = -0.0882

x[:,2,:,:] = (x[:,2,:,:]-MIN_r)/(MAX_r-MIN_r) # 200 Hz
x[:,3,:,:] = (x[:,3,:,:]-MIN_i)/(MAX_i-MIN_i)

MAX_r = 0.8407 # real part
MIN_r = -0.9845
MAX_i = 0.9071 # imaginary part
MIN_i = -1.0604

x[:,4,:,:] = (x[:,4,:,:]-MIN_r)/(MAX_r-MIN_r) # 796 Hz
x[:,5,:,:] = (x[:,5,:,:]-MIN_i)/(MAX_i-MIN_i)

MAX_r = 1.9075 # real part
MIN_r = -1.9146
MAX_i = 1.9392 # imaginary part
MIN_i = -1.8605

x[:,6,:,:] = (x[:,6,:,:]-MIN_r)/(MAX_r-MIN_r) # 12500 Hz
x[:,7,:,:] = (x[:,7,:,:]-MIN_i)/(MAX_i-MIN_i)

for i in range(0,num_test):

	x_test = x[i,:,:,:]

	img_tensor = x_test
	img_tensor.unsqueeze_(0) # add one dimension for the batch
	img_tensor = img_tensor.cuda()
	p_est,o1,o2,o3,o4 = model(Variable(img_tensor))
	# ==== prediction
	Y1 = o1.data.cpu().max(1)[1].numpy()
	Y2 = o2.data.cpu().max(1)[1].numpy()
	Y3 = o3.data.cpu().max(1)[1].numpy()
	Y4 = o4.data.cpu().max(1)[1].numpy()
	prediction = np.array([[Y1[0],Y2[0],Y3[0],Y4[0]]])
	prediction = prediction + 1

	print prediction
	
	result[i,:] = prediction

np.savetxt('example_outputs.txt', result, fmt = '%i')

