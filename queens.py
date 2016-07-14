"""
The prototypical backtracking problem is the classical n Queens Problem, 
first proposed by German chess enthusiast Max Bezzel in 1848 for the 
standard 8 x 8 board. The problem is to place n queens on an n x n 
chess board, so that no two queens can attack each other. For 
readers not familiar with the rules of chess, this means that no
two queens are in the same row, column, or diagonal.
"""

def queens(Q, r):
	"""
	Q is the array of length n that represents the chess board:
	each element of index i represent the position of the queen on
	ith row. r is an integer representing the fulfilled rows. 
	Initialized with 0.
	"""
	n = len(Q)
	if n < 4:
		raise AssertionError("Invalid input Q!")
	if r == n:
		return Q

	for i in range(n):
		flag = True
		for j in range(r):
			if Q[j] == i or Q[j] == r - Q[i] + j or Q[j] == r - Q[i] - j:
				flag = False
		if flag:
			Q[r] = i
			print Q
			queens(Q, r + 1)

	return Q

#print queens([-1]*4, 0)

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

