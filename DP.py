# A collection of python implementation for Dynamic Programming problems


"""
Maximum Value Contiguous Subsequence. Given a sequence 
of n real numbers A(1) ... A(n), determine a contiguous 
subsequence A(i) ... A(j) for which the sum of 
elements in the subsequence is maximized.
"""

def maximum_subseq(A):

	max_here, max_total = A[0], A[0]
	totalBgnIdx, totalEndIdx = -1, -1
	for i in range(1, len(A)):
		if A[i] > A[i] + max_here:
			max_here = A[i]
			hereBgnIdx = i
		else:
			max_here = A[i] + max_here

		if max_here > max_total:
			max_total = max_here
			totalBgnIdx = hereBgnIdx
			totalEndIdx = i

	return totalBgnIdx, totalEndIdx, max_total

#print maximum_subseq([-2, -5, 6, -2, 3, -10, 5, -6])
#print maximum_subseq([-2, 1, -3, 4, -1, 2, 1, -5, 4])
#print maximum_subseq([13, -3, -25, 20, -3, -16, -23, 18, 20, 
#	-7, 12, -5, -22, 15, -4, 7])

"""
Longest Increasing Subsequence. Given a sequence of n real
numbers A(1) ... A(n), determine a subsequence (not 
necessarily contiguous) of maximum length in which the 
values in the subsequence form a strictly increasing sequence.
"""

def longest_increasing_seq(A):

	D, T = [1] * len(A), [-1] * len(A)
	maxLen, maxIdx = 1, -1
	for i in range(1, len(A)):
		for j in range(i):
			if A[i] > A[j] and D[i] < D[j] + 1:
				D[i] = D[j] + 1
				T[i] = j
		if D[i] > maxLen:
			maxLen = D[i]
			maxIdx = i
	l = get_idx(maxIdx, T, [])
	return maxLen, [A[i] for i in l]

def get_idx(i, T, l):
	if T[i] == -1:
		l = [i] + l
		return l
	else:
		l = [i] + l
		return get_idx(T[i], T, l)

# print longest_increasing_seq([-2, -5, 6, -2, 3, -10, 5, -6])
# print longest_increasing_seq([-3, 1, 2, 4, -6, 5, 7, 2])
# print longest_increasing_seq([13, -3, -25, 20, -3, -16, -23, 18, 20, 
# 	-7, 12, -5, -22, 15, -4, 7])

"""
You are given n types of coin denominations of values 
v(1) < v(2) < ... < v(n) (all integers). Assume v(1) = 1, 
so you can always make change for any amount of money C. 
Give an algorithm which makes change for an amount 
of money C with as few coins as possible.
"""

def make_change(v, c):

	pass



