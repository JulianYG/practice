# -*- coding: utf-8 -*-
import random
import time
from matplotlib import pyplot as plt

"""
The prototypical backtracking problem is the classical n Queens Problem, 
first proposed by German chess enthusiast Max Bezzel in 1848 for the 
standard 8 x 8 board. The problem is to place n queens on an n x n 
chess board, so that no two queens can attack each other. For 
readers not familiar with the rules of chess, this means that no
two queens are in the same row, column, or diagonal.
"""

def queens(Q, r, s):
	"""
	Q is the array of length n that represents the chess board:
	each element of index i represent the position of the queen on
	ith row. 
	r is an integer representing the fulfilled rows, initialized with 0.
	"""
	n = len(Q)
	P = []
	for q in Q:
		P.append(q)

	for i in range(n):	# columns
		legal = True
		for j in range(r):	# filled rows
			if i == Q[j]:
				legal = False
			# Get rid of diagonal cases
			if i == Q[j] - r + j or i == Q[j] + r - j:
				legal = False
		if legal:
			P[r] = i
			queens(P, r + 1, s)
			if r == n - 1:	# only when the board is completely filled
				s.append(P)

# x = []
# queens([-1]*10, 0, x)
# print x

def fast_queen_search(n):
	"""
	A faster gradient heuristic search for N-queens problem;
	Only returns one solution.
	"""
	t, c = 1, 1
	Q = range(n)
	while c > 0: 
		random.shuffle(Q)
		c, t = 0, 0
		for i in range(n):
			for j in range(i + 1, n):
				if Q[i] + i == Q[j] + j or Q[i] - i == Q[j] - j: # on diagonal
					t += 1
		if t > 0:
			for i in range(n):
				for j in range(i + 1, n):
					P = []
					for q in Q:
						P.append(q)
					if need_swap(Q, i, j, t):
						s = Q[i]
						Q[i] = Q[j]
						Q[j] = s
						c += 1
	return Q

def need_swap(P, i, j, x):
	"""
	Check if swap Qi and Qj reduces total number of collision; if 
	it does then swap Qi and Qj.
	"""
	t = 0
	c = P[i]
	P[i] = P[j]
	P[j] = c
	for i in range(len(P)):
		for j in range(i + 1, len(P)):
			if P[i] + i == P[j] + j or P[i] - i == P[j] - j:
				t += 1
	if t < x:
		return True
	else:
		return False

# time1 = time.time()
# print fast_queen_search(17)
# time2 = time.time()

def comparison_perm_queen_search(n):
	t = 1
	Q = range(n)
	while t > 0:
		random.shuffle(Q)
		t = 0
		for i in range(n):
			for j in range(i + 1, n):
				if Q[i] + i == Q[j] + j or Q[i] - i == Q[j] - j: # on diagonal
					t += 1
	return Q

# time3 = time.time()
# print comparison_perm_queen_search(17)
# time4 = time.time()
# print time2 - time1, time4 - time3


def test_queens(n):
	timing = dict()
	original = []
	fast = []
	perm = []
	for i in range(4, n + 1):
		time1 = time.time()
		queens([-1]*i, 0, [])
		time2 = time.time()
		fast_queen_search(i)
		time3 = time.time()
		comparison_perm_queen_search(i)
		time4 = time.time()
		original.append(time2 - time1)
		fast.append(time3 - time2)
		perm.append(time4 - time3)
		# timing[i] = {'original': time2 - time1, 
		# 	'fast': time3 - time2, 'perm': time4 - time3}
	fig = plt.figure()
	# ax = fig.add_subplot(111)
	plt.plot(range(4, n + 1), original, label='original')
	plt.plot(range(4, n + 1), fast, label='fast')
	plt.plot(range(4, n + 1), perm, label='perm')
	plt.xlim(4, n + 1)
	plt.legend(loc='upper left', frameon=False)
	plt.yscale('log')
	plt.show()

# test_queens(15)

def subset_sum(S, x):

	if x == 0:
		return True, [[]]
	if len(S) == 0 and x != 0:
		return False, []
	
	res_exclusive = subset_sum(S[1:], x - S[0])
	res_inclusive = subset_sum(S[1:], x)

	if res_exclusive[0] and res_inclusive[0]:
		for item in res_exclusive[1]:
			item.append(S[0])
		return True, res_exclusive[1] + res_inclusive[1]

	if res_exclusive[0] or res_inclusive[0]:
		if not res_inclusive[0]:
			res_exclusive[1][0].append(S[0])
			return True, res_exclusive[1]
		else:
			return True, res_inclusive[1]

	return False, []

# print subset_sum([8,6,7,5,3,10,9], 15)
# print subset_sum([2,3], 5)
# print subset_sum([-2, 4, 8, -1, -3], -3)
# print subset_sum([42,2,3,1,-16,25,20,-13,8,18,6,9,4,7,-11,10,15,5,-9], 29)

def longest_accelerating_subsequence(prev, back, S):
	"""
	Call a sequence S accelerating if 2*S[i] < S[i−1] + S[i+1] for all i
	"""
	print S, prev, back
	if len(S) < 3:
		return 0
	else:
		curr = longest_accelerating_subsequence(prev, back, S[1:])

		if 2 * S[1] < prev + back:
			l = 2 + longest_accelerating_subsequence(S[0], S[2], S[1:])

			if l > curr:
				curr = l
		return curr

# print longest_accelerating_subsequence(10000, 10000, [2, 4, 8, 19, 37])


def recursive_longest_common_subsequence(s1, s2, cs):
	"""
	Very very inefficient recursive algorithm for LCS problem.
	"""
	if len(s1) == 0 or len(s2) == 0:
		return 0, cs

	else:
		if s1[0] == s2[0]:
			res = recursive_longest_common_subsequence(s1[1:], s2[1:], cs)
			newLen = res[0] + 1
			newStr = s1[0] + res[1]
			return newLen, newStr
		else:
			res1 = recursive_longest_common_subsequence(s1, s2[1:], cs)
			res2 = recursive_longest_common_subsequence(s1[1:], s2, cs)
			if res1[0] > res2[0]:
				return res1
			else:
				return res2

#print recursive_longest_common_subsequence('nematode knowledge', 'empty bottle', '')

def recursive_permutation(s):
	"""
	A recursive algorithm to generate all permutations of given string s.
	"""
	n = len(s)
	if n < 2:
		return set(s)
	else:
		S = set()
		for i in range(n):
			pivot = s[i]
			rest = s[:i] + s[i + 1:n]
			for p in recursive_permutation(rest):
				S.add(pivot + p)
		return S

#print recursive_permutation('abcd. Lol')

def recursive_longest_palindromic_subsequence(s):
	if len(s) < 2:
		return 1
	else:
		if s[0] == s[-1]:
			return recursive_longest_palindromic_subsequence(s[1:-1]) + 2
		else:
			return max(recursive_longest_palindromic_subsequence(s[:-2]), 
				recursive_longest_palindromic_subsequence(s[1:]))

#print recursive_longest_palindromic_subsequence('bbabcbcab')


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











