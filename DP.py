# A collection of python implementation for Dynamic Programming problems
# -*- coding: utf-8 -*- 
import numpy as np
import sys
import math
import operator

def maximum_subseq(A):
	"""
	Maximum Value Contiguous Subsequence. Given a sequence 
	of n real numbers A(1) ... A(n), determine a contiguous 
	subsequence A(i) ... A(j) for which the sum of 
	elements in the subsequence is maximized.
	"""

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

def longest_increasing_seq(A):
	"""
	Longest Increasing Subsequence. Given a sequence of n real
	numbers A(1) ... A(n), determine a subsequence (not 
	necessarily contiguous) of maximum length in which the 
	values in the subsequence form a strictly increasing sequence.
	"""

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
	"""
	Helper function for LIS backtrace.
	"""
	if T[i] == -1:
		l = [i] + l
		return l
	else:
		l = [i] + l
		return get_idx(T[i], T, l)

# print longest_increasing_seq([-2, -5, 6, -2, 3, -10, 5, -6])
# print longest_increasing_seq([-3, 1, 2, 4, -6, 5, 7, 2])
# print longest_increasing_seq([13, -3, -25, 20, -3, -16, -23, 18, 20, 
#	-7, 12, -5, -22, 15, -4, 7])


def make_change(v, C):
	"""
	You are given n types of coin denominations of values 
	v(1) < v(2) < ... < v(n) (all integers). Assume v(1) = 1, 
	so you can always make change for any amount of money C. 
	Give an algorithm which makes change for an amount 
	of money C with as few coins as possible.
	"""
	# initialize 
	w = [0] * (C + 1)
	for i in range(1, len(w)):
		w[i] = float('inf')

	for i in range(1, C + 1):
		for j in range(len(v)):
			if v[j] <= i:
				w[i] = min(w[i], w[i - v[j]] + 1)

	return w[-1]

# print make_change([1, 2, 3, 14], 117)

def longest_common_seq(s1, s2):
	"""
	The longest common subsequence (LCS) problem is the problem of 
	finding the longest subsequence common to all sequences in a 
	set of sequences (often just two sequences).
	"""
	m, n = len(s1), len(s2)
	w = np.zeros((m + 1, n + 1), dtype='int')
	common_subseq = ''

	for i in range(1, m + 1):
		for j in range(1, n + 1):
			if s1[i - 1] == s2[j - 1]:
				w[i][j] = 1 + w[i][j - 1]
			else:
				w[i][j] = max(w[i - 1][j], w[i][j - 1])
	
	# for backtrace
	while m > 0 and n > 0:

		if s1[m - 1] == s2[n - 1]:
			common_subseq = s1[m - 1] + common_subseq
			m -= 1
			n -= 1
		else:
			if w[m - 1][n] > w[m][n - 1]:
				m -= 1
			else:
				n -= 1

	return w[-1][-1], common_subseq

# print longest_common_seq('empty', 'pity')
# print longest_common_seq('s','d')
# print longest_common_seq('bcade', 'cebd')
# print longest_common_seq('nematode knowledge', 'empty bottle')

def edit_distance(s1, s2):
	"""
	Given two strings a and b on an alphabet Σ (e.g. the set of 
	ASCII characters, the set of bytes [0..255], etc.), the edit 
	distance d(a, b) is the minimum-weight series of edit operations 
	that transforms a into b. One of the simplest sets of edit 
	operations is:
	1. Insertion of a single symbol. If a = uv, then inserting the 
	symbol x produces uxv. This can also be denoted ε→x, using ε 
	to denote the empty string.
	2. Deletion of a single symbol changes uxv to uv (x→ε).
	3. Substitution of a single symbol x for a symbol y ≠ x 
	changes uxv to uyv (x→y).
	"""
	m, n = len(s1), len(s2)
	w = np.zeros((m + 1, n + 1), dtype='int')
	trace = []
	for i in range(m + 1):
		trace.append([])
		for j in range(n + 1):
			trace[i].append((0, 0))
	trace[0][0] = (-1, -1)

	# initialize the matrix
	for i in range(1, m + 1):
		w[i][0] = i
	for j in range(1, n + 1):
		w[0][j] = j

	# start iteration
	for i in range(1, m + 1):
		for j in range(1, n + 1):
			if s1[i - 1] == s2[j - 1]:
				dist = w[i - 1][j - 1]
			else: 
				dist = w[i - 1][j - 1] + 1
			w[i][j] = min(w[i - 1][j] + 1, w[i][j - 1] + 1, dist)
			# for backtrace
			if w[i][j] == w[i - 1][j] + 1:
				trace[i][j] = tuple((i - 1, j))
			elif w[i][j] == w[i][j - 1] + 1:
				trace[i][j] = tuple((i, j - 1))
			else:
				trace[i][j] = tuple((i - 1, j - 1))

	trace_idx = (m, n)
	modification = []
	trace_lst = [trace_idx]
	while trace[trace_idx[0]][trace_idx[1]] != (-1, -1):
		trace_lst.append(trace[trace_idx[0]][trace_idx[1]])
		trace_idx = trace[trace_idx[0]][trace_idx[1]]
	trace_lst.reverse()
	
	for curr in range(len(trace_lst) - 1):
		currIdx = trace_lst[curr]
		nextIdx = trace_lst[curr + 1]
		if nextIdx[0] > currIdx[0]:
			if nextIdx[1] > currIdx[1]:
				if s1[nextIdx[0] - 1] != s2[nextIdx[1] - 1]:
					modification.append('Substitute \'' + s1[nextIdx[0] - 1] + \
	 				'\' with \'' + s2[nextIdx[1] - 1] + '\'')
				else:
					modification.append('Equal: \'' + s1[nextIdx[0] - 1] + '\' , ' + s2[nextIdx[1] - 1] + '\'')
			else:
				modification.append('Delete \'' + s1[nextIdx[0] - 1] + '\'')
		elif nextIdx[1] > currIdx[1]:
			modification.append('Insert \'' + s2[nextIdx[1] - 1] + '\'')

	return w[-1][-1], modification

# print edit_distance('intention', 'execution')
# print edit_distance('ab', 'ba')
# print edit_distance('empt', 'my p')

def minimal_palindrome(p):
	"""
	Given a string, a partitioning of the string is a palindrome 
	partitioning if every substring of the partition is a palindrome. 
	For example, “aba|b|bbabb|a|b|aba” is a palindrome partitioning of 
	“ababbbabbababa”. Determine the fewest cuts needed for palindrome 
	partitioning of a given string. For example, minimum 3 cuts are needed 
	for “ababbbabbababa”. The three cuts are “a|babbbab|b|ababa”. 
	"""
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
	"""
	A helper recursive function to back trace the partitioned 
	palindromes that onstruct the given string.
	"""
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
	"""
	A simple function to check whether given string is a palindrome.
	"""
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
# print minimal_palindrome('x') #1

def longest_palindromic_subsequence(s):
	"""
	Given a sequence, find the length of the longest palindromic subsequence 
	in it. For example, if the given sequence is “BBABCBCAB”, then the output 
	should be 7 as “BABCBAB” is the longest palindromic subsequence in it. 
	“BBBBB” and “BBCBB” are also palindromic subsequences of the given sequence, 
	but not the longest ones.
	"""
	n = len(s)
	w = np.ones((n, n), dtype='int')

	for sublen in range(1, n + 1):
		for bgnIdx in range(n - sublen):
			endIdx = bgnIdx + sublen 
			if s[bgnIdx] == s[endIdx]:
				if sublen == 1:
					w[bgnIdx][endIdx] = 2
				else:
					w[bgnIdx][endIdx] = w[bgnIdx + 1][endIdx - 1] + 2
			else:
				w[bgnIdx][endIdx] = max(w[bgnIdx][endIdx - 1], w[bgnIdx + 1][endIdx])

	return w[0][-1]

# print longest_palindromic_subsequence('bbabcbcab')

def matrix_chain_order(p):
	"""
	Input: p[] = {40, 20, 30, 10, 30}   
	Output: 26000  
	There are 4 matrices of dimensions 40x20, 20x30, 30x10 and 10x30.
	Let the input 4 matrices be A, B, C and D. The minimum number of 
	multiplications are obtained by putting parenthesis in following way
	(A(BC))D --> 20*30*10 + 40*20*10 + 40*10*30
	"""
	n = len(p)
	w = np.zeros((n, n), dtype='int')
	for i in range(1, n):
		w[i][i] = 0

	for sublen in range(2, n + 1):
		for bgnIdx in range(1, n - sublen + 1):
			endIdx = bgnIdx + sublen - 1

			w[bgnIdx][endIdx] = sys.maxint

			for midIdx in range(bgnIdx, endIdx): 
				w[bgnIdx][endIdx] = min(w[bgnIdx][endIdx], 
					w[bgnIdx][midIdx] + w[midIdx+1][endIdx] + p[bgnIdx - 1] * p[midIdx] * p[endIdx])
	
	return w[1][-1]

#print matrix_chain_order([40, 20, 30, 10, 30])

def digit2int(d):
	"""
	A helper function to combine numbers in a list to decimal int value
	"""
	s = [str(c) for c in d]
	# To avoid overflow
	return 0 if not s else int(''.join(s))

def longest_increasing_digital_subsequence(D):
	"""
	Let D[1..n] be an array of digits, each an integer between 0 and 9. 
	A digital subsequence of D is an sequence of positive integers composed in 
	the usual way from disjoint substrings of D. 
	For example, 3, 4, 5, 9, 26, 35, 89, 93, 238, 462 is an increasing digital 
	subsequence with length 10 of the first several digits of π:
	3 1 4 1 5 9 2 6 5 3 5 8 9 7 9 3 2 3 8 4 6 2 6
	This function computes the longest increasing digital subsequence of D.
	"""
	A = [0] + D
	n = len(A)
	w = np.ones((n, n), dtype='int')
	# trackback matrix
	t = []
	for i in range(n):
		t.append([])
		for j in range(n):
			t[i].append((0, 0))
	# Mark as (-1, -1) for iteration base case
	t[0][0] = (-1, -1)

	# This is basically a upper-triangular matrix
	w[0, :] = 0
	w[:, 0] = 0

	# No need to account for the last one
	for sublen in range(1, n):
		for end_idx in range(sublen, n):

			start_idx = end_idx + 1 - sublen
			curr_val = digit2int(A[start_idx:end_idx + 1])

			prev_max = max(w[:sublen, start_idx - 1])
			prev_max_idx = (np.argmax(w[:sublen, start_idx - 1]), start_idx - 1)
			
			for jmp_idx in range(1, start_idx):
				
				if jmp_idx - sublen < 0:
					w[sublen, end_idx] = max(w[:sublen, jmp_idx]) + 1
					t[sublen][end_idx] = (np.argmax(w[:sublen, jmp_idx]), jmp_idx)
					continue;

				whole_val = digit2int(A[jmp_idx - sublen + 1: jmp_idx + 1])
			
				if curr_val > whole_val:

					together = max(w[sublen, jmp_idx] + 1, w[sublen, end_idx])
					separate = w[sublen - 1, end_idx - sublen] + 1

					if together > separate:	
						w[sublen, end_idx] = together

						if w[sublen, jmp_idx] + 1 >= w[sublen, end_idx]:
							t[sublen][end_idx] = (sublen, jmp_idx)
					else:
						w[sublen, end_idx] = separate
						t[sublen][end_idx] = (sublen - 1, end_idx - sublen)
	subseq_lst = []
	max_res_idx_set = np.argwhere(w == np.amax(w)).tolist()

	for max_res_idx in max_res_idx_set:
		num = []
		while max_res_idx != (-1, -1):
			sublen, col = max_res_idx[0], max_res_idx[1]
			max_res_idx = t[max_res_idx[0]][max_res_idx[1]]
			num.append(digit2int(A[col - sublen + 1: col + 1]))
		num.reverse()
		subseq_lst.append(num[1:])

	return np.amax(w), subseq_lst

# print longest_increasing_digital_subsequence([1,1,1,2,1])
# print longest_increasing_digital_subsequence([3,1,4,2,1])
# print longest_increasing_digital_subsequence([3,1,4,1,5,1,6])
# print longest_increasing_digital_subsequence([4,1,3,6])
# print longest_increasing_digital_subsequence([4,3,5,1])
# print longest_increasing_digital_subsequence([3,1,4,1,5,9,2,6,5])
# print longest_increasing_digital_subsequence([5,9,2,9,7,9,8,4,6])
# print longest_increasing_digital_subsequence([3,1,4,1,5,
#   	9,2,6,5,3,5,8,9,7,9,3,2,3,8,4,6,2,6])
# print longest_increasing_digital_subsequence([3,1,4,1,5,9,2, 
# 		6,5,3,5,8,9,7,9,3,2,3,8,4,6,2,6,4,3,3,8,3,2,7,9,5,2,0,8,8,4,1,9])

# l = ['3'] + list('141592653589793238462643383279502884197169399375105820974' + 
# 	'9445923078164062862089986280348253421170679821480865132823066470938446095505822317' + 
# 	'2535940812848111745028410270193852110555964462294895493038196442881097566593344612' + 
# 	'8475648233786783301194912983367336244065664308602139494639522473719070217986094370' + 
# 	'277053921717629317675238467481846766940513200056812714526356082778577134275778960917')

# print longest_increasing_digital_subsequence([int(i) for i in l])

def box_stacking(b):
	"""
	You are given a set of n types of rectangular 3-D boxes, where the i^th box 
	has length l(i), width w(i) and depth d(i) (all real numbers). You want to 
	create a stack of boxes which is as tall as possible, but you can only stack 
	a box on top of another box if the dimensions of the 2-D base of the lower 
	box are each strictly larger than those of the 2-D base of the higher box. 
	Of course, you can rotate a box so that any side functions as its base. It 
	is also allowable to use multiple instances of the same type of box.
	"""
	B = {}
	for dim in b:
		s_dim = np.sort(dim)
		B[(s_dim[0], s_dim[1], s_dim[2])] = s_dim[1] * s_dim[2]
		B[(s_dim[1], s_dim[0], s_dim[2])] = s_dim[0] * s_dim[2]
		B[(s_dim[2], s_dim[0], s_dim[1])] = s_dim[0] * s_dim[1]

	sorted_b = sorted(B.items(), key=operator.itemgetter(1))
	# initialize so that no consideration for orders
	D = [d[0][0] for d in sorted_b]
	for i in range(len(sorted_b)):
		for j in range(i):
			base_dim = sorted_b[i][0]
			top_dim = sorted_b[j][0]
			if base_dim[1] > top_dim[1] and base_dim[2] > top_dim[2]:
				if D[i] + top_dim[0] > D[i]:
					D[i] = D[j] + top_dim[0]
	return max(D)

# print box_stacking([[1,2,6], [5,3,1], [4,3,4], [8,2,7], [3,6,1], [2,5,9]])

def building_bridge(south, north):
	"""
	Consider a 2-D map with a horizontal river passing through its center. 
	There are n cities on the southern bank with x-coordinates a(1) ... a(n) 
	and n cities on the northern bank with x-coordinates b(1) ... b(n). 
	You want to connect as many north-south pairs of cities as possible with 
	bridges such that no two bridges cross. When connecting cities, you can 
	only connect city i on the northern bank to city i on the southern bank.
	"""


