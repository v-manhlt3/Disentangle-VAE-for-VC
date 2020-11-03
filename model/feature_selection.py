import torch
import numpy as np 
import torch.nn.functional as F 

class FeatureSelection(torch.nn.Module):
    
    def __init__(self, input_size, num_spk):

        super(FeatureSelection, self).__init__()
        self.input_size = input_size
        self.num_spk = num_spk
        layers = []
        self.ln1 = torch.nn.Linear(input_size, 1024)
        self.ln2 = torch.nn.Linear(1024, 1024)
        self.weight = torch.nn.Linear(1024, input_size)

        self.model = torch.nn.ModuleList(layers)
        self.cls_linear = torch.nn.Linear(input_size, 1024)
        self.prediction = torch.nn.Linear(1024, num_spk)
        self.softmax = torch.nn.Softmax(dim=-1)
    
    def forward(self, x):
        # x = out
        # for layers in self.model:
        #     print(layers)
        #     out = layers(x)
        out = torch.relu(self.ln1(x))
        out = torch.relu(self.ln2(out))
        out = torch.sigmoid(self.weight(out))
        return out

    def classify(self, x):

        prediction = torch.relu(self.cls_linear(x))
        prediction = self.prediction(prediction)
        prediction = self.softmax(prediction)
        return prediction

    def calc_loss(self, weight, label, prediction):

        # weight_loss = torch.mean(torch.sum(weight, dim=-1))
        CE_loss = torch.nn.functional.cross_entropy(prediction, label)
        return CE_loss

if __name__ == '__main__':
    data = torch.randn(10,512)

    model = FeatureSelection(2, 512, 109)
    print(model)
    weights = model(data)

    weighted_data = torch.mul(data, weights)
    prediction = model.classify(weighted_data)
    label = np.random.choice(109, 10)
    label = torch.from_numpy(label)
    print('label shape: ', label.shape)
    loss = model.calc_loss(weights, label, prediction)

    print(weights.shape)
    print(prediction.shape)
    print(prediction[0])
    print('Loss: ', loss.item())