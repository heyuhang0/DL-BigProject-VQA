# 50039 Deep Learning Big Project

# Dependency
```
matplotlib
pytorch
torchtext
torchvision
flask
easy-vqa
```

# Submition List
* `vqa2_attention_faster_rcnn.ipynb` create, training, loading trained model, testing the best model (Faster R-CNN + element-wise multiplication model trained on yes/no dataset)
* `vqa2_mul_vgg.ipynb`: create, training, loading trained model, testing the best model (VGG + element-wise multiplication model trained on yes/no dataset)
* `vqa2_mul_vgg.ipynb`: notebook to prepare and save the data for VQA2.0 dataset visualisation
* `dataset_visualization.xlsx`: VQA2.0 dataset visualisation graphs creation
* `utils/` folder: util files for model training related tasks, including creating training dataset and dataloader, training models, plotting model history, testing models
* `models/`: all the models we have experimented with, and selected model weights
* `demo-frontend/` demo front end code
* `demo.py` demo back end code
* all the model weights: saved in this onedrive link `https://sutdapac-my.sharepoint.com/:f:/g/personal/yuhang_he_mymail_sutd_edu_sg/EltdnfsWfZxOgapxoMhwIIUB8_Y5oIEuIv8GZ92FjVNMxg?e=mvuLM8`

# Run Instruction

## Models
We experimented on 9 models and training settings, and selected the best model and the best model with Attention mechanism to present the full training and testing process. 

**prerequisite: download all files in `https://sutdapac-my.sharepoint.com/:f:/g/personal/yuhang_he_mymail_sutd_edu_sg/EltdnfsWfZxOgapxoMhwIIUB8_Y5oIEuIv8GZ92FjVNMxg?e=mvuLM8` to `models/` folder**

run notebook `vqa2_mul_vgg.ipynb` to create, train on yes/no dataset, loading trained model weights, test the model with architecture VGG + element wise multiplication(best model).

run notebook `vqa2_attention_faster_rcnn.ipynb` to create, train on yes/no dataset, loading trained model weights, test the model with architecture Faster R-CNN + element wise multiplication (best model with attention mechanism.

## Dataset visualization
run notebook `VQA2_Dataset_Visualization.ipynb` to prepare and save the data for VQA2.0 dataset visualisation


## Run Demo Web App Locally

run `demo.py` in project root folder

run the following command in another terminal in the folder `demo-frontend`
```
yarn install
yarn start
```
go to `localhost:3000` to see the demo web app.
