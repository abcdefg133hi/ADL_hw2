import json
import os
import numpy as np
import matplotlib.pyplot as plt

epochs = 5

iteration = [1,2,3,4,5]
rouge1 = []
rouge2 = []
rougel = []

with open("learn.json", 'r') as fin:
    data = json.load(fin)



for epoch in range(epochs):
    r1 = data[f"Step{epoch+1}"]["Rouge-1"]
    r2 = data[f"Step{epoch+1}"]["Rouge-2"]
    rl = data[f"Step{epoch+1}"]["Rouge-l"]
    rouge1.append(r1)
    rouge2.append(r2)
    rougel.append(rl)


plt.plot(iteration, rouge1, label='rouge1', marker='o')
plt.plot(iteration, rouge2, label='rouge2', marker='o')
plt.plot(iteration, rougel, label='rougel', marker='o')
plt.title("Rouge Score vs Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Rouge Score")
plt.legend()
plt.savefig("LearningCurve.png")
#plt.show()


