import torch
import torch.nn as nn
import torch.nn.functional as F

class ContentLoss(nn.Module):
		def __init__(self, target):
			super().__init__()
			# we 'detach' the target content from the tree used
			# to dynamically compute the gradient: this is a stated value,
			# not a variable. Otherwise the forward method of the criterion
			# will throw an error.
			self.target = target.detach()#это константа. Убираем ее из дерева вычеслений
			self.loss = F.mse_loss(self.target, self.target )#to initialize with something
		def forward(self, input):
			self.loss = F.mse_loss(input, self.target)
			return input
			
def gram_matrix(input):
		batch_size, f_map_num, h, w = input.size()  # batch size(=1)
		# b=number of feature maps
		# (h,w)=dimensions of a feature map (N=h*w)
		# f_map_num = number of a feature map

		features = input.view(batch_size * f_map_num, h * w)  # resise F_XL into \hat F_XL

		G = torch.mm(features, features.t())  # compute the gram product

		# we 'normalize' the values of the gram matrix
		# by dividing by the number of element in each feature maps.
		return G.div(batch_size * h * w * f_map_num)
		
class StyleLoss(nn.Module):
		def __init__(self, target_feature, mask=None):
			super().__init__()
			if mask == None:
				self.mask = None
				self.target = gram_matrix(target_feature).detach() # это константа. Убираем ее из дерева вычеслений
			else: # Если маска задана, то нам нужно умножить обе матрицы на неё, а только потом искать матрицы Грама и лосс
				# Уменьшаем маску соответственно размерам target_feature
				mask_small = F.interpolate(mask, size=(target_feature.shape[2], target_feature.shape[3]))
				self.mask = mask_small.detach() # это константа. Убираем ее из дерева вычеслений
				self.target = gram_matrix(target_feature*self.mask).detach() # Матрицу Грама вычисляем по преобразованному изображению
			self.loss = F.mse_loss(self.target, self.target) # to initialize with something

		def forward(self, input):
			if self.mask != None:
				G = gram_matrix(input*self.mask) # Если на вход подали маску, то нужно умножить и входное изображение на неё
			else:
				G = gram_matrix(input)
			self.loss = F.mse_loss(G, self.target)
			return input
