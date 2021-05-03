# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn.utils.rnn as rnn_utils
import torchvision

# Create model
class VQAModel_ResNet_Concat(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, padding_idx:int):
        """
        VQA model with ResNet as image feature extraction method, 
        concatenation as image and text feature fusion method.

        @param vocab_size: int, vacabulary size of questions
        @param num_classes: int, possible answer output number
        @param padding_idx: int, index of padding token <pad> in question vocabulary 
        """
        super().__init__()
        # embedding layer
        self.rnn_embed = nn.Embedding(vocab_size, 512, padding_idx=padding_idx)
        
        # text feature extraction layers
        self.rnn = nn.GRU(512, 512)
        self.rnn_classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True))

        # image feature extraction layers
        self.cnn = torchvision.models.resnet50(pretrained=True)
        self.cnn.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Linear(1024, 256),
            nn.ReLU(True))
        
        # final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes))

    def forward(self, v, q, q_len):
        # text fearuture extraction, RNN
        embedded = self.rnn_embed(q)
        packed = rnn_utils.pack_padded_sequence(embedded, q_len)
        rnn_hideen = self.rnn(packed)[1].squeeze(0)
        rnn_out = self.rnn_classifier(rnn_hideen)

        # image feature extraction, CNN
        cnn_out = self.cnn(v)

        # Classifier
        features = torch.cat((rnn_out, cnn_out), 1)
        out = self.classifier(features)
        return out