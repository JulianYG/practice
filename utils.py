"""
Includes all utility functions needed for other coding exercises.
"""

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


def digit2int(d):
	"""
	A helper function to combine numbers in a list to decimal int value
	"""
	s = [str(c) for c in d]
	# To avoid overflow
	return 0 if not s else int(''.join(s))



