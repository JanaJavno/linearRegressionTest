import statsmodels.api as sm
import numpy as np
import pandas as pd

factorDetails, template = '', []
with open('dataset_task1.csv', 'r+', encoding='utf8') as fileinput:
    columns = fileinput.readline()
    columns = columns.strip('\n')
    factorDetails = columns.split(',')
    y = [factorDetails[0]]
    for i in range (factorDetails.__len__() - 1):
        template.append(factorDetails[i + 1])
    x = np.array([template])
    template = []

    for line in fileinput:
        line = line.strip('\n')
        factorDetails = line.split(',')
        y.append(float(factorDetails[0]))
        for i in range (factorDetails.__len__() - 1):
            template.append(float(factorDetails[i + 1]))
        x = np.append(x, [template], axis=0)
        template = []


df = pd.DataFrame(x[1:, 0:], columns=x[0, 0:])
target = pd.DataFrame(np.asarray(np.array(y)[1:]), columns=[y[0]])

X = df[['x_1', 'x_2', 'x_3', 'x_4']]
Y = target['y']

# Note the difference in argument order
model = sm.OLS(Y.astype(float), X.astype(float)).fit()
predictions = model.predict(X.astype(float)) # make the predictions by the model

# Print out the statistics
print(model.summary())