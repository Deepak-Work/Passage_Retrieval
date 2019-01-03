import numpy as np
def averaging(DL_file,BM25_file,output_file):
	fd = open(DL_file,'r',encoding="utf-8",errors = "ignore")
	fb = open(BM25_file,'r',encoding="utf-8",errors = "ignore")
	outputfile = open("answer.tsv", 'w', encoding="utf-8")
	i_count=0
	for l1 in fd:
	    i_count += 1
	    if(i_count%25000==0):
	        print(i_count)
	    d1 = [float(x) for x in l1.split("\t")]
	    uid1 = int(d1[0])
	    prob1 = np.array(d1[1:])
	    l2 = fb.readline()
	    d2 = [float(x) for x in l2.split("\t")]
	    uid2 = int(d2[0])
	    prob2 = np.array(d2[1:])
	    mean1 = np.mean(prob1)
	    mean2 = np.mean(prob2)
	    lambda1 = mean2/(mean2+mean1)
	    lambda2 = 1-lambda1
	    ans=str(uid1)
	    ansline = 0

	    for i in range(len(d1)-1):
	    	ansline = (lambda1*d1[i+1]+lambda2*d2[i+1])/2
	    	ans = ans + "\t" + str(ansline)
	    
	    outputfile.write(ans+"\n")

	fd.close()
	fb.close()
	outputfile.close()
DL_file = "answer(50).tsv"
BM25_file = "answer(46).tsv"
outputfile = "answer.tsv"
averaging(DL_file,BM25_file,outputfile)