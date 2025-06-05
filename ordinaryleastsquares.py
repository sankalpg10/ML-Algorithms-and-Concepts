

sales = [651,762,853,1062,1190,1293]
ads = [25,28,35,40,46,53]

sales_mean = sum(sales)/len(sales)
# print("sales_mean",sales_mean)
ads_mean = sum(ads)/len(ads)
# print("ads_mean",ads_mean)


b1_num = sum([(sales[i] - sales_mean)*(ads[i] - ads_mean) for i in range(len(sales))])
# print("b1_nu,",b1_num)
b1_den = [(ads[i] - ads_mean)**2 for i in range(len(sales))]
# print("b1_den",b1_den)
b1 = b1_num/sum(b1_den)



print("b1",b1)

b0 = sales_mean - (b1)*(ads_mean)

print("b0",b0)

y_pred = [(b1*(ads[i]) + (b0)) for i in range(len(sales))]

print("y_predicted: ",y_pred)