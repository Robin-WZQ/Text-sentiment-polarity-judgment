a = {"1":1,"2":2,"3":3,"4":4}
a1 = sorted(a.items(),key=lambda x:x[1],reverse=True)
b = {}
for i in a1:
    b[i[0]]=i[1]
print(b)