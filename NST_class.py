import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.models as models

import copy

import warnings
warnings.filterwarnings('ignore')


import losses # import StyleLoss and ContentLoss from a separate file


class NST:

	def __init__(self):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
		self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
		self.content_layers = ['conv_4'] # Список, показывающий после каких слоёв вставлять вычисление content loss
		self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'] # Список, показывающий после каких слоёв вставлять вычисление style loss
		#self.cnn = models.vgg19(pretrained=True).features.to(self.device).eval()
		self.cnn = torch.load('vgg19_cutted.pth').to(self.device).eval()

	##### Ниже идёт код, честно скопированный с домашки по NST #####
	#####(немного cкорректированный для использования в классе)#####
			
	class Normalization(nn.Module):
		def __init__(self, mean, std):
			super().__init__()
			# .view the mean and std to make them [C x 1 x 1] so that they can
			# directly work with image Tensor of shape [B x C x H x W].
			# B is batch size. C is number of channels. H is height and W is width.
			self.mean = torch.tensor(mean).view(-1, 1, 1)
			self.std = torch.tensor(std).view(-1, 1, 1)

		def forward(self, img):
			# normalize img
			return (img - self.mean) / self.std
			
	def get_style_model_and_losses(self, style_img_1, content_img,  style_img_2 = None, mask=None,
								   StyleLoss=losses.StyleLoss): # Добавим возможность подставить свой вариант лосса для стиля
		cnn = copy.deepcopy(self.cnn)

		# normalization module
		normalization = self.Normalization(self.cnn_normalization_mean, self.cnn_normalization_std).to(self.device)

		# just in order to have an iterable access to or list of content/syle
		# losses
		content_losses = []
		style_losses_1 = []
		style_losses_2 = [] # массив лоссов для второго стиля. Останется пустым, если стиль только один

		# assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
		# to put in modules that are supposed to be activated sequentially
		# Фактически, мы послойно собираем свою model, слои для которой мы берём из оригинальной модели cnn,
		# при этом мы задаём им свои имена и куда нужно вставляем вычисление loss функций.
		# Также ReLU мы заменяем на out-of-place версию (nn.ReLU(inplace=False))
		model = nn.Sequential(normalization)

		i = 0  # increment every time we see a conv
		for layer in cnn.children():
			if isinstance(layer, nn.Conv2d):
				i += 1
				name = 'conv_{}'.format(i)
			elif isinstance(layer, nn.ReLU):
				name = 'relu_{}'.format(i)
				# The in-place version doesn't play very nicely with the ContentLoss
				# and StyleLoss we insert below. So we replace with out-of-place
				# ones here.
				#Переопределим relu уровень
				layer = nn.ReLU(inplace=False)
			elif isinstance(layer, nn.MaxPool2d):
				name = 'pool_{}'.format(i)
			elif isinstance(layer, nn.BatchNorm2d):
				name = 'bn_{}'.format(i)
			else:
				raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

			model.add_module(name, layer)

			if name in self.content_layers:
				# add content loss:
				target = model(content_img).detach()
				content_loss = losses.ContentLoss(target)
				model.add_module("content_loss_{}".format(i), content_loss)
				content_losses.append(content_loss)

			if name in self.style_layers:
				# add style loss:
				# Добавляем слои в зависимости от того, какой вид Style Transfer мы используем
				if mask == None: # Если маска не используется, значит либо это NST с 1 стилем, либо с двумя, применяемыми ко всей картинке
					target_feature = model(style_img_1).detach()
					style_loss = StyleLoss(target_feature)
					model.add_module("style_loss1_{}".format(i), style_loss)
					style_losses_1.append(style_loss)
					if style_img_2 != None: # Если есть второй стиль, значит мы должны применить его одновременно с первым, добавляем соотв. лосс
						target_feature = model(style_img_2).detach()
						style_loss = StyleLoss(target_feature)
						model.add_module("style_loss2_{}".format(i), style_loss)
						style_losses_2.append(style_loss)
				else: # Если маска задана, значит это NST с двумя стилями, каждый из которых применяется к своей части изображения
					target_feature = model(style_img_1).detach()
					style_loss = StyleLoss(target_feature, mask)
					model.add_module("style_loss1_{}".format(i), style_loss)
					style_losses_1.append(style_loss)
					if style_img_2 != None: # Если есть второй стиль, значит мы должны применить его одновременно с первым, добавляем соотв. лосс
						target_feature = model(style_img_2).detach()
						style_loss = StyleLoss(target_feature, 1-mask) # Для 2-го стиля подаём обратную маску
						model.add_module("style_loss2_{}".format(i), style_loss)
						style_losses_2.append(style_loss)

		# now we trim off the layers after the last content and style losses
		# выбрасываем все уровни после последенего styel loss или content loss
		# Фактически, это делается потому что нам не интересен выход модели, нам важно лишь вычисление лоссов
		for i in range(len(model) - 1, -1, -1):
			if isinstance(model[i], losses.ContentLoss) or isinstance(model[i], StyleLoss):
				break
		model = model[:(i + 1)]

		return model, style_losses_1, style_losses_2, content_losses
		
	def get_input_optimizer(self, input_img):
		# this line to show that input is a parameter that requires a gradient
		# добавляет содержимое тензора катринки в список изменяемых оптимизатором параметров
		optimizer = optim.LBFGS([input_img.requires_grad_()]) 
		return optimizer
		
	def run_style_transfer(self, content_img, style_img_1, input_img, num_steps=500,
						style_weight=100000, style_1_factor=1, style_2_factor=1, # style_factor нужны для задания весов стилям 1 и 2
						content_weight=1, style_img_2=None, mask=None,
						StyleLoss=losses.StyleLoss):
		content_img = content_img.to(self.device, torch.float)
		style_img_1 = style_img_1.to(self.device, torch.float)
		input_img = input_img.to(self.device, torch.float)
		if style_img_2 != None:
			style_img_2 = style_img_2.to(self.device, torch.float)
		if mask != None:
			mask = mask.to(self.device, torch.float)
			
		"""Run the style transfer."""
		print('Building the style transfer model..')
		model, style_losses_1, style_losses_2, content_losses = self.get_style_model_and_losses(
			style_img_1, content_img, style_img_2, mask=mask, StyleLoss=StyleLoss)
		optimizer = self.get_input_optimizer(input_img)

		print('Optimizing..')
		run = [0]
		while run[0] <= num_steps:

			def closure():
				# correct the values 
				# это для того, чтобы значения тензора картинки не выходили за пределы [0;1]
				input_img.data.clamp_(0, 1)

				optimizer.zero_grad()

				model(input_img)

				style_1_score = 0
				style_2_score = 0
				content_score = 0

				for sl in style_losses_1:
					style_1_score += sl.loss
				for sl in style_losses_2:
					style_2_score += sl.loss
				for cl in content_losses:
					content_score += cl.loss
				
				#взвешивание ощибки
				style_1_score *= style_weight
				style_2_score *= style_weight
				content_score *= content_weight

				loss = style_1_score*style_1_factor + style_2_score*style_2_factor + content_score
				loss.backward()

				run[0] += 1
				if run[0] % 100 == 0:
					print("run {}:".format(run))
					if style_img_2 != None:
						print('Style 1 Loss : {:4f} Style 2 Loss : {:4f} Content Loss: {:4f}'.format(
							style_1_score.item(), style_2_score.item(), content_score.item()))
					else:
						print('Style Loss : {:4f} Content Loss: {:4f}'.format(
							style_1_score.item(), content_score.item()))
					#print()

				return style_1_score*style_1_factor + style_2_score*style_2_factor + content_score

			optimizer.step(closure)

		# a last correction...
		input_img.data.clamp_(0, 1)

		return input_img
