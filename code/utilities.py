

from numpy import reshape, identity, dot, array

def to_cvec(x):
	return reshape(x,(len(x),-1))

def permute_states(s):
	pi_s = array([
		[1,0,0,0],
		[0,0,1,0],
		[0,1,0,0],
		[0,0,0,1]
		]) 
	return dot(s, pi_s)