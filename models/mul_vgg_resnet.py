import torch
import torch.nn as nn
import torchvision.models as models


# image feature extracter using ResNet
class ImgEncoder_ResNet(nn.Module):
    def __init__(self, embed_size, pretrained=True):
        """
            VQA model image feature extraction using ResNet

            @param embed_size: int, number of features to extract in image feature extractor
            @param pretrained_img: bool, whether to use pretrained network for image feature extractor
        """
        super(ImgEncoder_ResNet, self).__init__()
        # load the pretrained model
        model = models.resnet152(pretrained=pretrained)
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


# image feature extracter using VGG
class ImgEncoder_VGG(nn.Module):
    def __init__(self, embed_size, pretrained=True):
        """
            VQA model image feature extraction using VGG

            @param embed_size: int, number of features to extract in image feature extractor
            @param pretrained_img: bool, whether to use pretrained network for image feature extractor
        """
        super(ImgEncoder_VGG, self).__init__()
        # load the pretrained model
        model = models.vgg19(pretrained=pretrained)
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
        # normalize the image feature vector
        l2_norm = img_feature.norm(p=2, dim=1, keepdim=True).detach()
        img_feature = img_feature.div(l2_norm)
        return img_feature


# test feature extracter
class QstEncoder(nn.Module):
    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):
        """
            VQA model text feature extraction model

            @param qst_vocab_size: int, vacabulary size of questions
            @param embed_size: int, number of features to extract in image and text feature extractor
            @param word_embed_size: int, word embedding size for text
            @param num_layers: int, number of recurrent layers in LSTM for text feature extractor
            @param hidden_size: int, number of features in the hidden state h in LSTM for text feature extractor
        """
        
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
        qst_feature = torch.cat((hidden, cell), 2)                    # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(0, 1)                     # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)                            # [batch_size, embed_size]

        return qst_feature


# VQA model
class VqaModel_Mul(nn.Module):
    # This VQA model is used for training all single-word-answer datasets

    def __init__(
        self, qst_vocab_size: int, ans_vocab_size: int, embed_size: int = 1024, 
        word_embed_size: int=300, num_layers: int=2, hidden_size: int=512,
        img_process="vgg", pretrained_img=True
    ):
        """
            VQA model with VGG/Resnet as image feature extraction method, 
            element-wise multiplication as image and text feature fusion method,
            multi-class classification. Work on both yes/no dataset, and single word answer dataset
        
            @param qst_vocab_size: int, vacabulary size of questions
            @param ans_vocab_size: int, possible answer output number
            @param embed_size: int, number of features to extract in image and text feature extractor
            @param word_embed_size: int, word embedding size for text
            @param num_layers: int, number of recurrent layers in LSTM for text feature extractor
            @param hidden_size: int, number of features in the hidden state h in LSTM for text feature extractor
            @param img_process: str, image feature extractor, "vgg" or "resnet"
            @param pretrained_img: bool, whether to use pretrained network for image feature extractor
        """
        super(VqaModel_Mul, self).__init__()
        # select image feature extractor according to imput param
        if img_process not in ["vgg", "resnet"]:
            raise TypeError("img_process shoud be 'vgg' or 'resnet")
        if img_process == 'vgg':
            self.img_encoder = ImgEncoder_VGG(embed_size, pretrained_img)
        else:
            self.img_encoder = ImgEncoder_ResNet(embed_size, pretrained_img)
        # text feature extractor
        self.qst_encoder = QstEncoder(qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size)
        # classification layers
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, qst):
        qst = torch.transpose(qst, 0, 1)
        # image features
        img_feature = self.img_encoder(img)                     # [batch_size, embed_size]
        # text features
        qst_feature = self.qst_encoder(qst)                     # [batch_size, embed_size]
        # element wise multiplication
        combined_feature = torch.mul(img_feature, qst_feature)  # [batch_size, embed_size]
        # classification: multiclass classification
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc1(combined_feature)           # [batch_size, ans_vocab_size]
        return combined_feature


class VqaModel_Mul_bin(nn.Module):
    # This VQA model is used for training only yes-no-answer datasets

    def __init__(
        self, qst_vocab_size:int, ans_vocab_size:int,embed_size:int=1024, 
        word_embed_size:int=300, num_layers:int=2, hidden_size:int=512,
        img_process="vgg", pretrained_img=True
    ):
        """
            VQA model with VGG/Resnet as image feature extraction method, 
            element-wise multiplication as image and text feature fusion method,
            binary class classification. Work on only yes/no dataset

            @param qst_vocab_size: int, vacabulary size of questions
            @param ans_vocab_size: int, possible answer output number
            @param embed_size: int, number of features to extract in image and text feature extractor
            @param word_embed_size: int, word embedding size for text
            @param num_layers: int, number of recurrent layers in LSTM for text feature extractor
            @param hidden_size: int, number of features in the hidden state h in LSTM for text feature extractor
            @param img_process: str, image feature extractor, "vgg" or "resnet"
            @param pretrained_img: bool, whether to use pretrained network for image feature extractor
        """
        super(VqaModel_Mul_bin, self).__init__()
        # select image feature extractor according to imput param
        if img_process not in ["vgg", "resnet"]:
            raise TypeError("img_process shoud be 'vgg' or 'resnet")
        if img_process == 'vgg':
            self.img_encoder = ImgEncoder_VGG(embed_size, pretrained_img)
        else:
            self.img_encoder = ImgEncoder_ResNet(embed_size, pretrained_img)
        # text feature extractor
        self.qst_encoder = QstEncoder(qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size)
        # classification layers
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, qst):
        qst = torch.transpose(qst, 0, 1)
        # image features
        img_feature = self.img_encoder(img)                     # [batch_size, embed_size]
        # text features
        qst_feature = self.qst_encoder(qst)                     # [batch_size, embed_size]
        # element wise multiplication
        combined_feature = torch.mul(img_feature, qst_feature)  # [batch_size, embed_size]
        # classification: binary classification
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc1(combined_feature)           # [batch_size, ans_vocab_size=1000]
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        # binary classification
        combined_feature = self.sigmoid(combined_feature)
        return combined_feature
