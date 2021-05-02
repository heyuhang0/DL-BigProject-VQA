
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class VNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = fasterrcnn_resnet50_fpn(pretrained=pretrained).backbone
        self.out_channels = self.backbone.out_channels

    def forward(self, images):
        return self.backbone(images)['3']


class QNet(nn.Module):
    def __init__(self, vocab, embedding_dim, out_dim, weights_path=None):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), embedding_dim)
        self.tanh = nn.Tanh()
        self.rnn = nn.LSTM(embedding_dim, out_dim)

        if weights_path:
            counter = 0
            weights = self.embedding.weight.detach().numpy()
            with open(weights_path, encoding='utf-8') as f:
                for line in f:
                    elements = line.split(' ')

                    word = elements[0]
                    if word not in vocab.stoi:
                        continue

                    embed = np.asarray(elements[1:], dtype='float32')
                    weights[vocab.stoi[word]] = embed

                    counter += 1
                    if counter / len(vocab) > 0.9:
                        break
            self.embedding.weight.data.copy_(torch.from_numpy(weights))

    def forward(self, q, q_len):
        embedded = self.embedding(q)
        tanhed = self.tanh(embedded)
        packed = rnn_utils.pack_padded_sequence(tanhed, q_len)
        _, (_, features) = self.rnn(packed)
        features = features.squeeze(0)
        return features


class VQANet(nn.Module):
    def __init__(self, vocab, num_classes: int, glove_path='./glove.6B.100d.txt'):
        super().__init__()
        self.v_net = VNet()
        self.q_net = QNet(vocab, 100, 256, glove_path)

        self.v_query = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.Sigmoid())
        self.q_query = nn.Sequential(
            nn.Linear(256, 128),
            nn.Sigmoid())
        self.attention_softmax = nn.Softmax(dim=1)

        self.q_fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.Sigmoid())

        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes))
        self.last_attention = None

    def forward(self, v, q, q_len):
        v_feat = self.v_net(v)
        q_feat = self.q_net(q, q_len)

        v_query = self.v_query(v_feat)
        q_query = self.q_query(q_feat)
        attention = (v_query * q_query.view((-1, 128, 1, 1))).sum(dim=1)
        attention = self.attention_softmax(attention.view(-1, 7*7)).view(-1, 1, 7, 7)
        self.last_attention = attention.detach()  # save for visualization

        v_final = (v_feat * attention).view((-1, 256, 7*7)).sum(dim=2)
        q_final = self.q_fc(q_feat)
        out = self.classifier(v_final * q_final)
        return out
