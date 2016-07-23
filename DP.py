# A collection of python implementation for Dynamic Programming problems
# -*- coding: utf-8 -*- 
import numpy as np

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
# 	-7, 12, -5, -22, 15, -4, 7])


def make_change(v, c):
	"""
	You are given n types of coin denominations of values 
	v(1) < v(2) < ... < v(n) (all integers). Assume v(1) = 1, 
	so you can always make change for any amount of money C. 
	Give an algorithm which makes change for an amount 
	of money C with as few coins as possible.
	"""
	

	pass


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
	A helper recursive fcunction to back trace the partitioned 
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


def minimum_palindromic_insertions(s):
	"""
	Given a string, find the minimum number of characters to be inserted 
	to convert it to palindrome. 
	Returns the minimum number of characters and the modified string of palindrome.
	"""
	# base case
	if len(s) < 2:
		return 0, s
	# if begin and end as palindrome, only need to check the rest
	if s[0] == s[-1]:
		minLen, palStr = minimum_palindromic_insertions(s[1:-1])
		return minLen, s[0] + palStr + s[0]
	else:
		restLeft = s[1:]
		minLenLeft, palStrLeft = minimum_palindromic_insertions(restLeft)
		if s[0] != restLeft[-1]:
			minLenLeft += 1
			palStrLeft = s[0] + palStrLeft + s[0]
		# try both left and right ends, and pick the best
		revS = s[::-1]
		restRight = revS[1:]
		minLenRight, palStrRight = minimum_palindromic_insertions(restRight)
		if revS[0] != restRight[-1]:
			minLenRight += 1
			palStrRight = (revS[0] + palStrRight + revS[0])[::-1]
		if minLenLeft < minLenRight:
			return minLenLeft, palStrLeft
		else:
			return minLenRight, palStrRight

# print minimum_palindromic_insertions('bboocb')

def matrix_chain_order(d):
	pass

