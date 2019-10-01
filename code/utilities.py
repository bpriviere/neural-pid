

from numpy import reshape,identity,dot,array,sqrt,pi,exp


def eval_normal_prob(prob,x):
	# normal distribution
	mu = prob.loc.detach()
	std = prob.scale.detach()
	var = std**2
	return 1/sqrt(2*pi*var)*exp(-((x-mu)**2)/(2*var))

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