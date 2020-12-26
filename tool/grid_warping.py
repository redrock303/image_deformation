import torch
import numpy as np 

def Backward(tensorInput, tensorFlow):
	Backward_tensorGrid = {}
	if str(tensorFlow.size()) not in Backward_tensorGrid:
		tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
		tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))
		print('tensorFlow',tensorFlow.shape)
		Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()
	# end
	print('Backward_tensorGrid',Backward_tensorGrid)
	tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)
	# print('torch.abs y sum',torch.abs(tensorFlow[:,1]).sum())
	# print('torch.abs x sum',torch.abs(tensorFlow[:,0]).sum())
	# # tensorFlow[:,1]=0
	# print('torch.abs 1',torch.abs(tensorFlow[:,1]).sum())
	# print('tensorInput',tensorInput.shape,)             # zero deform with flow

	return torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
def wrapImg(img,flow):
	''' img   h w 3 
		flow  h w 2
	'''  
	img = img /255.0 
	imgTensor = np.transpose(img,(2,0,1))
	flowTensor = np.transpose(flow,(2,0,1))
	imgTensor = torch.from_numpy(imgTensor).float().cuda().unsqueeze(0)
	flowTensor = torch.from_numpy(flowTensor).float().cuda().unsqueeze(0)
	result = Backward(imgTensor,flowTensor).cpu().numpy()[0]

	result = np.transpose(result,(1,2,0))
	result = (result*255.0).astype(np.uint8)
	return result
    # print('flowTensor',result.shape)
    # input('check')

# img = np.zeros((57,62,3),dtype=np.uint8)
# flow = np.random.rand(57,62,2)
# wrapImg(img,flow)