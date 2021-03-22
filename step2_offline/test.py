ratio = 0.045
cunru_by_year = 1294*12
print(cunru_by_year)
quchu_by_year = 6000 + 8000
total = 0
print('total zhuniancunru={}'.format(cunru_by_year*15))
for i in range(15):
    total = cunru_by_year*(1+ratio) + total*(1+ratio)
    if i not in [0, 1,2,3]:
        total = total - quchu_by_year
print('total with licai={}'.format(total))