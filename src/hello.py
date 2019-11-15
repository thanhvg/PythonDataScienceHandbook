import numpy as np
import pandas as pd
import matplotlib
import seaborn

plt = matplotlib.pyplot
seaborn.set()
# show as window pop up with TkAgg backend
matplotlib.use('TkAgg')


L = np.array([1, 2, 3])

x = np.arange(1, 6)
np.add.reduce(x)

x = np.arange(5)
y = np.empty(5)
np.multiply(x, 10, out=y)

y = np.zeros(10)
np.power(2, x, out=y[::2])

L = np.random.random(100)

data = pd.read_csv('../notebooks/data/president_heights.csv')
heights = np.array(data['height(cm)'])


plt.hist(heights)
plt.title('US Pres heights')
plt.xlabel('Height cm')
plt.ylabel('number')
plt.show()

a = np.array([0, 1, 2])
b = np.array([5, 5, 5])
a + b

x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 50)[:, np.newaxis]
z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

plt.imshow(z, origin='lower', extent=[0, 5, 0, 5], cmap='viridis')
plt.show()

rainfall = pd.read_csv('../notebooks/data/Seattle2014.csv')['PRCP'].values

inches = rainfall / 254
inches.shape

plt.hist(inches, 40)
plt.show()
inches > 4

mean = [0, 2]
cov = [[1, 2], [2,5]]
rand = np.random.RandomState(42)
X = rand.multivariate_normal(mean, cov, 100)
X.shape

plt.scatter(X[:, 0], X[:,1])
plt.show()

indices = np.random.choice(X.shape[0], 20, replace=False)
selection = X[indices]


plt.scatter(X[:, 0], X[:, 1], alpha=0.3)
plt.scatter(selection[:, 0], selection[:, 1],  s=200)
plt.show()

x = np.arange(10)
i = np.array([2, 1, 8, 4])
x[i] = 99

x = np.zeros(10)
np.add.at(x, [1, 2, 3], 1)

np.random.seed(42)
x = np.random.randn(100)

bins = np.linspace(-5, 5, 20)

counts = np.zeros_like(bins)

i = np.searchsorted(bins, x)
np.add.at(counts, i, 1)
# XXX plt deprecates passing drawstyle
plt.plot(bins, counts, linestyle='steps')
plt.show()

x = rand.rand(10, 2)
plt.scatter(x[:, 0], x[:, 1], s=100)
plt.show()

dist_sq = np.sum((x[:, np.newaxis, :] - x[np.newaxis, :, :]) ** 2, axis=-1)
