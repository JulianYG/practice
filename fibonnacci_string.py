
def find_char(s0, s1, n, k):
	"""
	Given two base strings s0, s1, find the kth character
	of the nth fibonnacci string constructed by the two 
	strings. E.g:
	s0 = "aaa", s1 = "bcdef"
	n = 5, k = 12
	aaa, bcdef, aaabcdef, bcdefaaabcdef, aaabcdefbcdefaaabcdef
	The 12th character of the 5th string is 'e'.
	"""
	if n == 1:
		if k < 1 or k > len(s0):
			raise AssertionError("Invalid input k!")
		return s0[k - 1]
	if n == 2:
		if k < 1 or k > len(s1):
			raise AssertionError("Invalid input k!")
		return s1[k - 1]

	a, b = len(s0), len(s1)
	for i in range(n - 2):
		c = a
		a = b 
		b = c + b

	if k > b - a:
		return find_char(s0, s1, n - 1, k - (b - a))
	else:
		return find_char(s0, s1, n - 2, k)

def test_find_char():
	"""
	Test function for corner cases
	"""
	test_one = ''
	for j in range(1, 13):
		test_one += find_char('ab', 'cogna', 4, j)
	if test_one != 'cognaabcogna':
		print 'Test 1 Failed; Got ' + test_one + ', Expecting ' + 'cognaabcogna'
	else:
		print 'Passed Test 1'
	test_two = ''
	for j in range(1, 15):
		test_two += find_char('cdk', 'l', 6, j)
	if test_two != 'lcdklcdkllcdkl':
		print 'Test 2 Failed; Got ' + test_two + ', Expecting ' + 'lcdklcdkllcdkl'
	else:
		print 'Passed Test 2'
	test_three = ''
	for j in range(1, 6):
		test_three += find_char('a', 'b', 5, j)
	if test_three != 'abbab':
		print 'Test 3 Failed; Got ' + test_three + ', Expecting ' + 'abbab'
	else:
		print 'Passed Test 3'

test_find_char()

