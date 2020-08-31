import csv
import sys
from operator import itemgetter

lowrows=int(sys.argv[1])
highrows=int(sys.argv[2])

pi_given_S={}


with open('synthetic_real_P_genomes.csv', newline='') as SNP_probs_file:
	reader=csv.DictReader(SNP_probs_file)
	rowcount=0
	for row in reader:
		if rowcount<lowrows:
			rowcount+=1
			continue
		elif rowcount>=lowrows and rowcount<=highrows:
			rowcount+=1
		else:
			break
		pi_given_S[row['ID']]={'F': float(row['F']), 'M': float(row['M']), 'Blue': float(row['Blue']), 'Int': float(row['Int']), 'Brown': float(row['Brown']), 'Blonde': float(row['Blonde']), 'Brownh': float(row['Brownh']), 'Black': float(row['Black']), 'Pale' : float(row['Pale']), 'Intskin' : float(row['Intskin']), 'Darkskin' : float(row['Darkskin'])}


S_given_X={}

with open('synthetic_predicted_phenotypes.csv','r') as csvf:
	reader=csv.DictReader(csvf)
	rowcount=0
	for row in reader:
		if rowcount<lowrows:
			rowcount+=1
			continue
		elif rowcount>=lowrows and rowcount<=highrows:
			rowcount+=1
		else:
			break
		X=row['ID']

		for S in pi_given_S.keys():

			p=pi_given_S[S][row['SEX']]*pi_given_S[S][row['HCOLOR']]*pi_given_S[S][row['SKINCLR']]*pi_given_S[S][row['EYECLR']]

			if X in S_given_X:
				S_given_X[X][S]=p
			else:
				S_given_X[X]={S:p}

fn=0
fp=0
tp=0
tn=0
count=0

#threshold=float(sys.argv[3])
maxval=0
for key, value in S_given_X.items():
	for k,v in sorted(value.items(), key=itemgetter(1), reverse=True)[:int(sys.argv[3])]:
	# for k,v in sorted(value.items(), key=itemgetter(1), reverse=True):
	# 	if v>=maxval:
	# 		maxval=v
	# 	if v>=threshold:
		if k==key:
			count+=1
	# 			tp+=1
	# 		else:
	# 			fp+=1
	# 	else:
	# 		if k==key:
	# 			fn+=1
	# 		else:
	# 			tn+=1
	# 		#print(k,v, sorted(value.items(), key=itemgetter(1), reverse=True).index((k,v)))

# tp=count
# fp=(len(S_given_X.items())*int(sys.argv[3]))-count
# fn=(len(S_given_X.items())-count)
# tn=(len(S_given_X.items())**2-tp-fp-fn)

# #print(tp, fp, tn, fn)
# #print('\n\n')
# print(fp/(fp+tn), tp/(tp+fn))
# #print(tn/(tn+fp), fn/(fn+tp))


print(count)