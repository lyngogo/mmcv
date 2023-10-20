n,k = map(int,input().split())
a = list(map(int,input().split()))
def numOfmod2(a):
    count = 0
    while a%2==0:
        count+=1
        a = a/2
    return count