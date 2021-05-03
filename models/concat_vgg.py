# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn.utils.rnn as rnn_utils
import torchvision

# Create model
class VQAModel_VGG_Concat(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, padding_idx: int):
        super().__init__()
        self.rnn_embed = nn.Embedding(vocab_size, 512, padding_idx=padding_idx)
        self.rnn = nn.GRU(512, 512)
        self.rnn_classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True))
        self.cnn = torchvision.models.vgg16(pretrained=True)
        self.cnn.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 256))
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes))
        

    def forward(self, v, q):
        embedded = self.rnn_embed(q)
        rnn_out = self.rnn(embedded)[1].squeeze(0)
        cnn_out = self.cnn(v)
        features = torch.cat((rnn_out, cnn_out), 1)
        return self.classifier(features)

    def forward(self, v, q, q_len):
        # RNN
        embedded = self.rnn_embed(q)
        packed = rnn_utils.pack_padded_sequence(embedded, q_len)
        rnn_hideen = self.rnn(packed)[1].squeeze(0)
        rnn_out = self.rnn_classifier(rnn_hideen)

        # CNN
        cnn_out = self.cnn(v)

        # Classifier
        features = torch.cat((rnn_out, cnn_out), 1)
        out = self.classifier(features)
        return out

# 
# model = VQAModel_VGG16_(len(question_vocab), len(answer_vocab), PAD_IDX)