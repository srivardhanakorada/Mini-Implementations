import torch

'''
Broadcasting Rules:
a) Move shapes to right completly
b) If matched no issue, else one of them have to be 1 and non-one value will be there in the result
c) If not there, the value which is there in the other tensor will be there in the result
'''
A = torch.rand(32,1,28,1)
B = torch.rand(32,3,1,28)

C = torch.add(A,B)
print(C.shape) # 32 * 3 * 28 * 28