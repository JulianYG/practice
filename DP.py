# A collection of python implementation for Dynamic Programming problems


"""
Maximum Value Contiguous Subsequence. Given a sequence 
of n real numbers A(1) ... A(n), determine a contiguous 
subsequence A(i) ... A(j) for which the sum of 
elements in the subsequence is maximized.
"""

def maximum_subseq(A):

	max_here, max_total = 0, 0
	begin, end = 0, 0
	for i in range(len(A)):
		if max_here < max_here + A[i]:
			end = i		
		max_total = max(max_here, A[i] + max_here)
		max_here = max(max_here, A[i])
		if max_here < max_total:
			begin = i
	return begin, end, max_total

#print maximum_subseq([-2, -5, 6, -2, 3, -10, 5, -6])


"""
Longest Increasing Subsequence. Given a sequence of n real
numbers A(1) ... A(n), determine a subsequence (not 
necessarily contiguous) of maximum length in which the 
values in the subsequence form a strictly increasing sequence.
"""

def longest_increasing_seq(A):

	T= [1] * len(A)
	for i in range(1, len(A)):
		for j in range(i):
			if A[i] > A[j]:
				T[i] = max(T[i], T[j] + 1)

	return max(T)

print longest_increasing_seq([-2, -5, 6, -2, 3, -10, 5, -6])

"""
You are given n types of coin denominations of values 
v(1) < v(2) < ... < v(n) (all integers). Assume v(1) = 1, 
so you can always make change for any amount of money C. 
Give an algorithm which makes change for an amount 
of money C with as few coins as possible.
"""

def make_change(v, c):

	pass



