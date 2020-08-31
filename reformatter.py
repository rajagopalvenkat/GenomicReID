import sys

if sys.argv[1]=='a':
	for line in open('temp', 'r').readlines():
		if line[0].isdigit():
			print(line.replace('\n', ','), end='')
		else:
			print(line, end='')

elif sys.argv[1]=='b':
	for line in open('temp_init', 'r').readlines():
		print(line[:-2])
	print()