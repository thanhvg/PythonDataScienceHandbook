import numpy as np
import pandas as pd
import matplotlib
import seaborn

plt = matplotlib.pyplot
seaborn.set()

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

matplotlib.use('TkAgg')
plt.hist(heights)
plt.title('US Pres heights')
plt.xlabel('Height cm')
plt.ylabel('number')
plt.show()
