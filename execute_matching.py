import os
import sys

n=[(10*k)-1 for k in range(1, 456//10 + 2)]

os.system('rm temp')

for win in n:
	lst=[10*k for k in range((456-win)//10 + 1)]
	lst.append(max(456-win, 0))
	
	for i in lst:
		os.system('python3 quickmatch.py ' + str(i) + " " + str(i+win) + " " + str(1) + " >> temp")
	os.system("echo >> temp")
	for i in lst:
		os.system('python3 quickmatch.py ' + str(i) + " " + str(i+win) + " " + str(3) + " >> temp")
	os.system("echo >> temp")
	for i in lst:
		os.system('python3 quickmatch.py ' + str(i) + " " + str(i+win) + " " + str(5) + " >> temp")
	os.system("echo >> temp")

os.system('python3 reformatter.py a > temp_init')
os.system('python3 reformatter.py b > temp_final')
os.system('python3 calc.py temp_final > results')
os.system('rm temp_init temp temp_final')
