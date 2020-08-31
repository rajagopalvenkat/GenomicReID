import numpy as np
import sys

f=open(sys.argv[1], 'r')
k=0

a0=[]
a1=[]
a2=[]

xc=0

for line in f.readlines():
	if xc==138:
		print(a0)
		print(a1)
		print(a2)
		a0=[]
		a1=[]
		a2=[]
		xc=0
		k=0

	x=np.array([int(y) for y in line.rstrip('\n').split(',')])
	if k==0:
		a0+=[np.mean(x),]
		k+=1
		k=k%3
	elif k==1:
		a1+=[np.mean(x),]
		k+=1
		k=k%3
	else:
		a2+=[np.mean(x),]
		k+=1
		k=k%3
	xc+=1

