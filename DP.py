# A collection of python implementation for Dynamic Programming problems

import numpy as np

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


def longest_common_seq(s1, s2):

	pass
	
def edit_distance(s1, s2):

	pass

def minimal_palindrome(p):

	n = len(p)
	w = np.ones((n, n), dtype='int')
	div = []

	for sublen in range(1, n + 1):
		for bgnIdx in range(n - sublen + 1):
			endIdx = bgnIdx + sublen - 1
			if isPalindrome(p[bgnIdx:endIdx+1]):
				w[bgnIdx][endIdx] = 1
			else:
				w[bgnIdx][endIdx] = sublen
				for midIdx in range(bgnIdx, endIdx):
					if w[bgnIdx][endIdx] >= w[bgnIdx][midIdx] + w[midIdx+1][endIdx]:
						div.append(((bgnIdx, midIdx), (midIdx+1, endIdx)))
						w[bgnIdx][endIdx] = w[bgnIdx][midIdx] + w[midIdx+1][endIdx]
	div.reverse()
	return w[0][-1], track_palindrome((0, len(p)-1), div, w, p)


def track_palindrome(i, I, W, P):
	if W[i] == 1:
		return [P[i[0]:i[1]+1]]
	else:
		for it in I:
			leftIdx, rightIdx = it[0], it[1]
			if leftIdx[0] == i[0] and rightIdx[1] == i[1]:
				l = track_palindrome(leftIdx, I, W, P)
				r = track_palindrome(rightIdx, I, W, P)
				return l + r

def isPalindrome(s):

	if len(s) < 2:
		return True
	else:
		if s[0] == s[-1]:
			return isPalindrome(s[1:-1])
		return False

# print minimal_palindrome('bubbaseesabanana')	#3
# print minimal_palindrome('baseesab') #1
# print minimal_palindrome('ooxx')	#2
# print minimal_palindrome('ababbbabbababa') #4
# print minimal_palindrome('axffde')	#5
# print minimal_palindrome('oxxo')	#1


