import torch
import torch.nn as nn
import torchvision.models as models


class ImgEncoder_ResNet(nn.Module):
    def __init__(self, embed_size):
        super(ImgEncoder_ResNet, self).__init__()
        # load the pretrained model
        model = models.resnet152(pretrained=True)
        in_features = model.fc.in_features
        # replace the final fully connected layer so that the output dimension is embedding size
        model.fc = nn.Linear(in_features, embed_size)
        self.model = model

    def forward(self, image):
        with torch.no_grad():
            img_feature = self.model(image)
        # normalize the image feature vector
        l2_norm = img_feature.norm(p=2, dim=1, keepdim=True).detach()
        img_feature = img_feature.div(l2_norm)
        return img_feature


class ImgEncoder_VGG(nn.Module):
    def __init__(self, embed_size):
        super(ImgEncoder_VGG, self).__init__()
        # load the pretrained model
        model = models.vgg19(pretrained=True)
        in_features = model.classifier[-1].in_features
        # remove the last layer(fc) and design customized layer
        model.classifier = nn.Sequential(
            *list(model.classifier.children())[:-1])
        self.model = model
        self.fc = nn.Linear(in_features, embed_size)

    def forward(self, image):
        with torch.no_grad():
            img_feature = self.model(image)
        img_feature = self.fc(img_feature)
        l2_norm = img_feature.norm(p=2, dim=1, keepdim=True).detach()
        img_feature = img_feature.div(l2_norm)
        return img_feature


class QstEncoder(nn.Module):
    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):
        super(QstEncoder, self).__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2*num_layers*hidden_size, embed_size)  # 2 for hidden and cell states

    def forward(self, question):
        qst_vec = self.word2vec(question)                             # [batch_size, max_qst_length=30, word_embed_size=300]
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)                             # [max_qst_length=30, batch_size, word_embed_size=300]
        _, (hidden, cell) = self.lstm(qst_vec)                        # [num_layers=2, batch_size, hidden_size=512]
        # element-wise multiplication
        qst_feature = torch.cat((hidden, cell), 2)                    # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(0, 1)                     # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)                            # [batch_size, embed_size]

        return qst_feature


class VqaModel_Mul(nn.Module):
    # This VQA model is used for training all single-word-answer datasets

    def __init__(self, embed_size, qst_vocab_size, ans_vocab_size, word_embed_size, num_layers, hidden_size, img_process="vgg"):

        super(VqaModel_Mul, self).__init__()
        if img_process not in ["vgg", "resnet"]:
            raise TypeError("img_process shoud be 'vgg' or 'resnet")
        if img_process == 'vgg':
            self.img_encoder = ImgEncoder_VGG(embed_size)
        else:
            self.img_encoder = ImgEncoder_VGG(embed_size)
        self.qst_encoder = QstEncoder(qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, qst):
        qst = torch.transpose(qst, 0, 1)
        img_feature = self.img_encoder(img)                     # [batch_size, embed_size]
        qst_feature = self.qst_encoder(qst)                     # [batch_size, embed_size]
        combined_feature = torch.mul(img_feature, qst_feature)  # [batch_size, embed_size]
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc1(combined_feature)           # [batch_size, ans_vocab_size]
        # training on all single-word-answer questions,
        # so the output dimension size is ans_vocab_size instead of a single score value
        return combined_feature


class VqaModel_Mul_bin(nn.Module):
    # This VQA model is used for training only yes-no-answer datasets

    def __init__(self, embed_size, qst_vocab_size, ans_vocab_size, word_embed_size, num_layers, hidden_size, img_process="vgg"):

        super(VqaModel_Mul_bin, self).__init__()
        if img_process not in ["vgg", "resnet"]:
            raise TypeError("img_process shoud be 'vgg' or 'resnet")
        if img_process == 'vgg':
            self.img_encoder = ImgEncoder_VGG(embed_size)
        else:
            self.img_encoder = ImgEncoder_VGG(embed_size)
        self.qst_encoder = QstEncoder(qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, qst):
        qst = torch.transpose(qst, 0, 1)
        img_feature = self.img_encoder(img)                     # [batch_size, embed_size]
        qst_feature = self.qst_encoder(qst)                     # [batch_size, embed_size]
        combined_feature = torch.mul(img_feature, qst_feature)  # [batch_size, embed_size]
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc1(combined_feature)           # [batch_size, ans_vocab_size=1000]
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        # training on yes-no-answer datasets,
        # model would output the score value between 0 and 1
        combined_feature = self.sigmoid(combined_feature)
        return combined_feature
