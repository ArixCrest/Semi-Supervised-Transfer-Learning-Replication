# -*- coding: utf-8 -*-
"""AML_Minor1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yb4k8iFRl3UADCKFfpDqTxh1IM2VKrMu
"""

import plotly.graph_objs as go
import plotly.io as pio

!pip install -U kaleido

accuracies = []
with open('cifar10.txt') as f:
    values = f.readlines()
    for id in range(len(values)):
        values[id] = values[id].split()
        for j in range(len(values[id])):
            values[id][j] = float(values[id][j])
    accuracies = values

print("maximum accuracies for CIFAR-10 dataset with 200 labelled datapoints.")
for i in range(len(accuracies)):
    print(max(accuracies[i]))

trace1 = go.Scatter(x = [(i+1) for i in range(len(accuracies[0]))],y = accuracies[0], name='Without Consistency Loss')
trace2 = go.Scatter(x = [(i+1) for i in range(len(accuracies[0]))],y = accuracies[1], name='AKC consistency = 0.5')
trace3 = go.Scatter(x = [(i+1) for i in range(len(accuracies[0]))],y = accuracies[2], name='AKC and ARC consistenc6 = 0.5')


data = [trace1,trace2, trace3]

layout = go.Layout(title='Test Accuracies on CIFAR-10 Dataset')

fig = go.Figure(data=data, layout=layout)

fig.show()
pio.write_image(fig, 'cifar-10.png')

accuracies = []
with open('svhn_1000.txt') as f:
    values = f.readlines()
    for id in range(len(values)):
        values[id] = values[id].split()
        for j in range(len(values[id])):
            values[id][j] = float(values[id][j])
    accuracies = values

print("maximum accuracies for SVHN dataset with 1000 labelled datapoints.")
for i in range(len(accuracies)):
    print(max(accuracies[i]))

trace1 = go.Scatter(x = [(i+1) for i in range(len(accuracies[0]))],y = accuracies[0], name='Without Consistency Loss')
trace2 = go.Scatter(x = [(i+1) for i in range(len(accuracies[0]))],y = accuracies[1], name='AKC consistency = 0.5')
trace3 = go.Scatter(x = [(i+1) for i in range(len(accuracies[0]))],y = accuracies[2], name='AKC and ARC consistenc6 = 0.5')


data = [trace1,trace2, trace3]

layout = go.Layout(title='Test Accuracies on SVHN Dataset(1000 labelled data)')

fig = go.Figure(data=data, layout=layout)

fig.show()
pio.write_image(fig, 'svhn_1000.png')

accuracies = []
with open('svhn_2000.txt') as f:
    values = f.readlines()
    for id in range(len(values)):
        values[id] = values[id].split()
        for j in range(len(values[id])):
            values[id][j] = float(values[id][j])
    accuracies = values

print("maximum accuracies for SVHN dataset with 2000 labelled datapoints.")
for i in range(len(accuracies)):
    print(max(accuracies[i]))

trace1 = go.Scatter(x = [(i+1) for i in range(len(accuracies[0]))],y = accuracies[0], name='Without Consistency Loss')
trace2 = go.Scatter(x = [(i+1) for i in range(len(accuracies[0]))],y = accuracies[1], name='AKC consistency = 0.5')
trace3 = go.Scatter(x = [(i+1) for i in range(len(accuracies[0]))],y = accuracies[2], name='AKC and ARC consistenc6 = 0.5')


data = [trace1,trace2, trace3]

layout = go.Layout(title='Test Accuracies on SVHN Dataset(2000 labelled data)')

fig = go.Figure(data=data, layout=layout)

fig.show()
pio.write_image(fig, 'svhn_2000.png')