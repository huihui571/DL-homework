import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import transforms, models
from PIL import Image
import argparse
import numpy as np
import os
 
use_gpu = torch.cuda.is_available()# 是否能够使用GPU
#use_gpu = False
print(torch.cuda.get_device_name(0))
print(use_gpu)
dtype = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor  #判断数据允许GPU 否则CPU tensor
#dtype = torch.FloatTensor

#导入数据
def load_image(image_path, transforms=None, max_size=None, shape=None):
	image = Image.open(image_path)#打开图片  image_path 为图片路径
	image_size = image.size #图片大小
 
	if max_size is not None:
		#获取图像size，为sequence
		image_size = image.size
		#转化为float的array
		size = np.array(image_size).astype(float)
		size = max_size / size * size
		image = image.resize(size.astype(int), Image.ANTIALIAS)
		#Image.ANTIALIAS在当前的PIL版本中，这个滤波器只用于改变尺寸和缩略图方法。
	if shape is not None: image = image.resize(shape, Image.LANCZOS)#Image.LANCZOS 缩小图片比例
	#必须提供transform.ToTensor，转化为4D Tensor .unsqueeze(0)表示增维 .squeeze(0)为降维
	if transforms is not None: image = transforms(image).unsqueeze(0) #是否拷贝到GPU
	return image.type(dtype)


class VGGNet(nn.Module): 
	def __init__(self):
		super(VGGNet, self).__init__()
		self.select = ['0', '5', '10', '19', '28']#设置需要提取的
		self.vgg19 = models.vgg19(pretrained = True).features 

	def forward(self, x): 
		features = [] #name类型为str，x为Variable 
		for name, layer in self.vgg19._modules.items(): 
			x = layer(x) 
			if name in self.select: 
				features.append(x) 

		return features


def main(config): 
	#定义图像变换操作，必须定义.ToTensor()。 
	transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) ]) 
	#加速content和style图像，style图像resize成同样大小 content.size(0-3)：0表示batch 1表示通道，2，3表示图片大小 

	content = load_image(config.content, transform, max_size = config.max_size)
	style = load_image(config.style, transform, shape = [content.size(2), content.size(3)])
	#.clone()将concent复制一份作为target，并需要计算梯度，作为最终的输出

	target = Variable(content.clone(), requires_grad = True) 
	optimizer = torch.optim.Adam([target], lr = config.lr, betas=[0.5, 0.999]) 

	vgg = VGGNet() 
	if use_gpu: 
		vgg = vgg.cuda() 
	for step in range(config.total_step): #分别计算5个特征图 
		target_features = vgg(target) 
		content_features = vgg(Variable(content)) 
		style_features = vgg(Variable(style)) 

		content_loss = 0.0 
		style_loss = 0.0 

		for f1, f2, f3 in zip(target_features, content_features, style_features): #计算content_loss
			content_loss += torch.mean((f1 - f2)**2) # **表示上标 **2 表示平方 
			n, c, h, w = f1.size() #将特征reshape成二维矩阵相乘，求gram矩阵 
			f1 = f1.view(c, h * w) 
			f3 = f3.view(c, h * w) 
			f1 = torch.mm(f1, f1.t())#f1.t() 表示f1的转置，但不改变f1
			f3 = torch.mm(f3, f3.t())
			#计算style_loss 
			style_loss += torch.mean((f1 - f3)**2) / (c * h * w)

		#计算总的loss
		loss = content_loss + style_loss * config.style_weight
		#反向求导与优化
		optimizer.zero_grad()
		loss.backward(retain_graph=True)
		optimizer.step()

		if (step+1) % config.log_step == 0:
			print ('Step [%d/%d], Content Loss: %.4f, Style Loss: %.4f' %(step+1, config.total_step, content_loss.item(), style_loss.item()))
		if (step+1) % config.sample_step == 0:
			# Save the generated image
			denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
			img = target.clone().cpu().squeeze()
			img = denorm(img.data).clamp_(0, 1) #为什么要归到0~1?
			torchvision.utils.save_image(img, './imgout/output-%d.png' %(step+1))


if __name__ == "__main__": 
	parser = argparse.ArgumentParser() 
	parser.add_argument('--content', type=str, default='content.jpg')
	parser.add_argument('--style', type=str, default='style.jpg')
	parser.add_argument('--max_size', type=int, default=400)
	parser.add_argument('--total_step', type=int, default=500)
	parser.add_argument('--log_step', type=int, default=10)
	parser.add_argument('--sample_step', type=int, default=10)
	parser.add_argument('--style_weight', type=float, default=100) 
	parser.add_argument('--lr', type=float, default=0.003)
	config = parser.parse_args() 
	print(config) 
	main(config)
