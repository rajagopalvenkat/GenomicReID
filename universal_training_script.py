import os
import time

os.system('mkdir retrain_univ_models')

for eps in [0.001, 0.005, 0.01, 0.025, 0.05]:
	os.system('cp sexVGG.pth retrain_univ_models/sexVGG_' + str(eps) +  '.pth')
	os.system('cp skinVGG.pth retrain_univ_models/skinVGG_' + str(eps) +  '.pth')
	os.system('cp eyecolorVGG.pth retrain_univ_models/eyecolorVGG_' + str(eps) +  '.pth')
	os.system('cp hcolorVGG.pth retrain_univ_models/hcolorVGG_' + str(eps) +  '.pth')
	for asd in range(5):
		for x in ['a','b','c','d','e']:
			os.system("echo Epsilon : " + str(eps) + ", Iter : " + x)
			os.system('python3 generate_adversarial_examples.py ' + str(eps) + ' 2 ' + x)
		os.system('python3 universal_advTraining_iteration.py ' + str(eps) + ' sex')
		os.system('python3 universal_advTraining_iteration.py ' + str(eps) + ' skin')
		os.system('python3 universal_advTraining_iteration.py ' + str(eps) + ' hcolor')
		os.system('python3 universal_advTraining_iteration.py ' + str(eps) + ' eyecolor')
		time.sleep(20)	# For GPU/CPU cooldown. Adjust according to preferred thermal performance.
	time.sleep(100)		# For GPU/CPU cooldown. Adjust according to preferred thermal performance.
