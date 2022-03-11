import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# read in the data file
df = pd.read_csv(r'D:\Jessica\Documents\School\Project Winter\Jessica_Feb3\SaltSugar Data Compilation.csv', skiprows=3)

# import the data for a given frequency
f = 2450
findex = df.loc[df["fMHz"] == f]

# import the distilled water data for the frequency given above
waterdc = findex["water_dc"].values
waterdl = findex["water_dl"].values

# import the salt data for the frequency given above
salt2dc = findex["salt2_dc"].values
salt2dl = findex["salt2_dl"].values

salt4dc = findex["salt4_dc"].values
salt4dl = findex["salt4_dl"].values

salt6dc = findex["salt6_dc"].values
salt6dl = findex["salt6_dl"].values

salt8dc = findex["salt8_dc"].values
salt8dl = findex["salt8_dl"].values

salt10dc = findex["salt10_dc1"].values
salt10dl = findex["salt10_dl1"].values

# import the sugar data for the frequency given above
sugar2dc = findex["sugar2_dc"].values
sugar2dl = findex["sugar2_dl"].values

sugar4dc = findex["sugar4_dc"].values
sugar4dl = findex["sugar4_dl"].values

sugar6dc = findex["sugar6_dc"].values
sugar6dl = findex["sugar6_dl"].values

sugar8dc = findex["sugar8_dc"].values
sugar8dl = findex["sugar8_dl"].values

sugar10dc = findex["sugar10_dc"].values
sugar10dl = findex["sugar10_dl"].values

# prepare data sets to plot
salt = np.array([2.00, 4.00, 6.00, 8.00, 10.00]).reshape(-1, 1)
saltdc = np.concatenate([salt2dc, salt4dc, salt6dc, salt8dc, salt10dc])
saltdl = np.concatenate([salt2dl, salt4dl, salt6dl, salt8dl, salt10dl])

sugar = np.array([2.00, 4.00, 6.00, 8.00, 10.00]).reshape(-1, 1)
sugardc = np.concatenate([sugar2dc, sugar4dc, sugar6dc, sugar8dc, sugar10dc])
sugardl = np.concatenate([sugar2dl, sugar4dl, sugar6dl, sugar8dl, sugar10dl])

# split the data randomly for training and testing
sample_size = 0.33
salt_random = 1
sugar_random = 1
salt_train, salt_test, saltdc_train, saltdc_test = train_test_split(salt, saltdc, test_size=sample_size, random_state=salt_random)
salt_train2, salt_test2, saltdl_train, saltdl_test = train_test_split(salt, saltdl, test_size=sample_size, random_state=salt_random)

salt_train, salt_test, saltdc_train, saltdc_test = train_test_split(salt, saltdc, test_size=sample_size, random_state=salt_random)
salt_train2, salt_test2, saltdl_train, saltdl_test = train_test_split(salt, saltdl, test_size=sample_size, random_state=salt_random)

sugar_train, sugar_test, sugardc_train, sugardc_test = train_test_split(sugar, sugardc, test_size=sample_size, random_state=sugar_random)
sugar_train2, sugar_test2, sugardl_train, sugardl_test = train_test_split(sugar, sugardl, test_size=sample_size, random_state=sugar_random)

sugar_train, sugar_test, sugardc_train, sugardc_test = train_test_split(sugar, sugardc, test_size=sample_size, random_state=sugar_random)
sugar_train2, sugar_test2, sugardl_train, sugardl_test = train_test_split(sugar, sugardl, test_size=sample_size, random_state=sugar_random)

saltdcmodel = LinearRegression().fit(salt_train, saltdc_train)
print("Dielectric Constant Equation (Salt): ", saltdcmodel.coef_ , "x + ", saltdcmodel.intercept_)
print("R2 (Training Set): ", saltdcmodel.score(salt_train, saltdc_train))
print("R2 (Test Set): ", saltdcmodel.score(salt_test, saltdc_test))

saltdlmodel = LinearRegression().fit(salt_train2, saltdl_train)
print("Dielectric Loss Equation (Salt): ", saltdlmodel.coef_ , "x + ", saltdlmodel.intercept_)
print("R2 (Training Set): ", saltdlmodel.score(salt_train2, saltdl_train))
print("R2 (Test Set): ", saltdlmodel.score(salt_test2, saltdl_test))

sugardcmodel = LinearRegression().fit(sugar_train, sugardc_train)
print("Dielectric Constant Equation (Sugar): ", sugardcmodel.coef_ , "x + ", sugardcmodel.intercept_)
print("R2 (Training Set): ", sugardcmodel.score(sugar_train, sugardc_train))
print("R2 (Test Set): ", sugardcmodel.score(sugar_test, sugardc_test))

sugardlmodel = LinearRegression().fit(sugar_train2, sugardl_train)
print("Dielectric Loss Equation (Sugar): ", sugardlmodel.coef_ , "x + ", sugardlmodel.intercept_)
print("R2 (Training Set): ", sugardlmodel.score(sugar_train2, sugardl_train))
print("R2 (Test Set): ", sugardlmodel.score(sugar_test2, sugardl_test))

# perform salt or sugar content prediction based on dc_test data and compare with salt_test or sugar_test
print("Salt Content Prediction using Test Data")
saltpredict = []
saltpredict2 = []
for i in range(0, len(saltdc_test)):
    saltpredict.append((saltdc_test[i] - saltdcmodel.intercept_)/saltdcmodel.coef_)
    saltpredict2.append((saltdl_test[i] - saltdlmodel.intercept_) / saltdlmodel.coef_)
    print("Measured Salt: ", salt_test[i], "Predicted Salt from DC: ", saltpredict[i], "Predicted Salt from DL: ", saltpredict2[i])

    i += 1

print("Sugar Content Prediction using Test Data")
sugarpredict = []
sugarpredict2 = []
for i in range(0, len(saltdc_test)):
    sugarpredict.append((sugardc_test[i] - sugardcmodel.intercept_) / sugardcmodel.coef_)
    sugarpredict2.append((sugardl_test[i] - sugardlmodel.intercept_) / sugardlmodel.coef_)
    print("Measured Sugar: ", sugar_test[i], "Predicted Sugar from DC: ", sugarpredict[i], "Predicted Sugar from DL: ", sugarpredict2[i])

    i += 1

# plot the models, training sets, and test sets
content_ = np.linspace(0, 10, 100)

plot1 = plt.figure(1)
plt.scatter(salt_train, saltdc_train, label="Training Set")
plt.scatter(salt_test, saltdc_test, label="Test Set")
plt.plot(content_, saltdcmodel.coef_*content_ + saltdcmodel.intercept_, label="Model")

handles, labels = plt.gca().get_legend_handles_labels()
order = [1, 2, 0]
plt.legend([handles[i] for i in order], [labels[i] for i in order])

plt.xlabel("Salt Content (%)")
plt.ylabel("Dielectric Constant ($\epsilon^{'}$)")

plot2 = plt.figure(2)
plt.scatter(salt_train2, saltdl_train, label="Training Set")
plt.scatter(salt_test2, saltdl_test, label="Test Set")
plt.plot(content_, saltdlmodel.coef_*content_ + saltdlmodel.intercept_, label="Model")

handles, labels = plt.gca().get_legend_handles_labels()
order = [1, 2, 0]
plt.legend([handles[i] for i in order], [labels[i] for i in order])

plt.xlabel("Salt Content (%)")
plt.ylabel("Dielectric Loss ($\epsilon^{''}$)")

plot3 = plt.figure(3)
plt.scatter(sugar_train, sugardc_train, label="Training Set")
plt.scatter(sugar_test, sugardc_test, label="Test Set")
plt.plot(content_, sugardcmodel.coef_*content_ + sugardcmodel.intercept_, label="Model")

handles, labels = plt.gca().get_legend_handles_labels()
order = [1, 2, 0]
plt.legend([handles[i] for i in order], [labels[i] for i in order])

plt.xlabel("Sugar Content (%)")
plt.ylabel("Dielectric Constant ($\epsilon^{'}$)")

plot4 = plt.figure(4)
plt.scatter(sugar_train2, sugardl_train, label="Training Set")
plt.scatter(sugar_test2, sugardl_test, label="Test Set")
plt.plot(content_, sugardlmodel.coef_*content_ + sugardlmodel.intercept_, label="Model")

handles, labels = plt.gca().get_legend_handles_labels()
order = [1, 2, 0]
plt.legend([handles[i] for i in order], [labels[i] for i in order])

plt.xlabel("Sugar Content (%)")
plt.ylabel("Dielectric Loss ($\epsilon^{''}$)")

plt.show()