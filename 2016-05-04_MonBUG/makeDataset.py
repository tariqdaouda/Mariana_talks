from pyGeno.Genome import *
import random, cPickle

def getRandomSeq(l):
	n = ['A', 'T', 'C', 'G']
	seq = []
	for i in xrange(l) :
		seq.append(random.choice(n))

	return ''.join(seq)

def encode(seq) :
	code = {
		'A' : [0, 0, 0, 1],
		'T' : [0, 0, 1, 0],
		'C' : [0, 1, 0, 0],
		'G' : [1, 0, 0, 0]
	}
	
	res = []

	for c in seq :
		res.extend(code[c])

	return res

ref = Genome(name = 'GRCh37.75_Y-Only')
seqs = []
targs = []
for trans in ref.iterGet(Transcript) :
	if len(trans.cDNA) >= 150 :
		r = random.randint(0, len(trans.cDNA) - 100)
		dnaseq = trans.cDNA[r: r + 100]
		seqs.append(encode(dnaseq))
		targs.append(1)
		randomseq = getRandomSeq(100)
		seqs.append(encode(randomseq))
		targs.append(0)

t = int(len(seqs) * 0.2) 
validation = [seqs[0:t], targs[0:t]]
train = [seqs[t:], targs[t:]]
print seqs[0:10]
cPickle.dump([train, validation], open('Ydna.pkl', 'wb'))
