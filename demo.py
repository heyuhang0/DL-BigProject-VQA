import base64
import io
import json
import string
from collections import Counter
from typing import Callable

import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
from flask import Flask, jsonify, request
from PIL import Image, ImageDraw
from torchtext.vocab import Vocab
from torchvision import transforms


def load_ResNet50FPN_GloveLSTM_attention(device='cpu') -> Callable[[Image.Image, str], str]:
    from model_ResNet50FPN_GloveLSTM_attention import VQANet

    device = torch.device(device)

    with open('vocab_question.json', 'r', encoding='utf-8') as fp:
        question_freqs = json.load(fp)
    with open('vocab_answer.json', 'r', encoding='utf-8') as fp:
        answer_freqs = json.load(fp)

    question_vocab = Vocab(Counter(question_freqs), specials=['<pad>', '<unk>'])
    answer_vocab = Vocab(Counter(answer_freqs), specials=['<unk>'], min_freq=10)

    model = VQANet(question_vocab, len(answer_vocab), None)
    state_dict = torch.load('model_ResNet50FPN_GloveLSTM_attention.pth')
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    image_transform = transforms.Compose([
        transforms.Resize(int(224 / 0.875)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    def tokenizer(question: str):
        # to lower case
        question = question.lower()
        # remove punctuation
        trans = str.maketrans('', '', string.punctuation)
        question = question.translate(trans)
        # split words
        return question.split()

    def question_transform(question: str):
        tokens = tokenizer(question)
        indices = [question_vocab[token] for token in tokens]
        return torch.tensor(indices, dtype=torch.long)

    def predict(image: Image.Image, question: str) -> str:
        with torch.no_grad():
            # preprocess
            image = image_transform(image)
            question = question_transform(question)
            question_length = len(question)
            # to batch
            images = torch.stack([image], 0)
            question_lengths = [question_length]
            questions = rnn_utils.pad_sequence([question])
            # predict
            output = model(images.to(device), questions.to(device), question_lengths)
            output = torch.argmax(output)
            # convert transformed image back to PIL image
            image = image.detach().cpu().numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean
            image = np.clip(image, 0, 1)
            image = Image.fromarray(np.uint8(image*255))
            # visualize attention
            draw = ImageDraw.Draw(image, 'RGBA')
            attention = model.last_attention.cpu().numpy().reshape((7, 7))
            attention -= np.min(attention)
            attention /= np.max(attention)
            for x in range(7):
                for y in range(7):
                    pt1 = (x*32, y*32)
                    pt2 = (x*32+32, y*32+32)
                    draw.rectangle((pt1, pt2), fill=(255, 0, 0, int(200 * attention[y][x])))
            # export image
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='JPEG')
            image_b64 = base64.b64encode(image_bytes.getvalue())
            image_uri = 'data:image/jpeg;base64,' + image_b64.decode('ascii')

            return [image_uri, answer_vocab.itos[output.item()]]

    return predict


app = Flask(__name__)

model_loaders = {
    'attention_all': load_ResNet50FPN_GloveLSTM_attention


}

models = {}


@ app.route('/api/vqa', methods=['POST'])
def vqa():
    # Read image
    try:
        if 'image' not in request.json:
            raise ValueError('`image` not found in the request form')
        image_uri = request.json['image']
        _, image_b64 = image_uri.split(',', 1)
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        msg = f'failed to read image due to error: {e}'
        app.logger.error(msg)
        return jsonify({'message': msg}), 400

    # Read question
    question = request.json.get('question', 'what is this?')

    # Get model
    model_name = request.json.get('model', 'attention_all')
    if model_name in model_loaders:
        if model_name not in models:
            models[model_name] = model_loaders[model_name]()
        model = models[model_name]
    else:
        msg = f'unknown model: {model_name}'
        app.logger.error(msg)
        return jsonify({'message': msg}), 404

    # Answer question
    answer = model(image, question)

    app.logger.info(f'Q: {question} A: {answer[-1]}')
    return jsonify({
        'question': question,
        'answer': answer
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
