import sys
sys.stdout=open("test.txt", "w")
print("hello")
sys.stdout.close()

sys.stdout=open("test_2.txt", "w")
print("hello")
sys.stdout.close()