import random, sys
random.seed(int(sys.argv[1]) if len(sys.argv)>1 else 0)
T = int(sys.argv[2]) if len(sys.argv)>2 else 1000
A=[[0.7,0.05,0.25],[0.1,0.8,0.1],[0.2,0.3,0.5]]
B=[[0.7,0.2,0.1,0.0],[0.1,0.4,0.3,0.2],[0.0,0.1,0.2,0.7]]
def draw(p):
    r=random.random();s=0
    for i,x in enumerate(p):
        s+=x
        if r<s: return i
    return len(p)-1
x=0;obs=[]
for t in range(T):
    obs.append(draw(B[x])); x=draw(A[x])
print("3 3 0.54 0.26 0.20 0.19 0.53 0.28 0.22 0.18 0.60")
print("3 4 0.50 0.20 0.11 0.19 0.22 0.28 0.23 0.27 0.19 0.21 0.15 0.45")
print("1 3 0.3 0.2 0.5")
print(T," ".join(map(str,obs)))
