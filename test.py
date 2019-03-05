class A:
	def __init__(self, a):
		self.a = a
		self.b = 5
	
def main():
	a = A(2)
	print(a.a)
	print(a.b)

if __name__ == '__main__':
	main()
