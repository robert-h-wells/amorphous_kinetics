names = [str(273),str(1),str(0),str(35)]


x = ['kmc',*names,'.txt']
y = print(*x, sep='_')

z = '_'.join([str(v) for v in x])
print(z)
