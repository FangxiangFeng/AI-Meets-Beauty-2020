import sys
import os
import csv
from tqdm import tqdm
import numpy as np
import pickle
import warnings

import torch
import torchvision
import torchvision.transforms as transforms
# import torchvision.models as models

from training_dataset import retrieval_dataset
from pooling import *
from net import resnet50bt, resnet101bt

PROGRAM_DIR = os.path.dirname(os.path.abspath(__file__))

transform_480 = transforms.Compose([
    transforms.Resize((480,480)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

def load_feature(feat_name):
    with open(feat_name, "rb") as file_to_read:
        feature=pickle.load(file_to_read)
    name=feature['name']
    return name,feature

if __name__ == "__main__":
    test_image_path=sys.argv[1]
    result_path=sys.argv[2]
    
    # test_image_path='./Challenge_validation_set_2020/val_2020'
    # result_path='./result/result_resnet50101bt_mac.csv'

    res50bt_path = os.path.join(PROGRAM_DIR, './pretrained/resnet50bt.pth.tar')
    res101bt_path = os.path.join(PROGRAM_DIR, './pretrained/resnet101bt.pth.tar')
    res50bt_feature_path= os.path.join(PROGRAM_DIR, './feature/feat_resnet50bt_mac.pkl')
    res101bt_feature_path=os.path.join(PROGRAM_DIR, './feature/feat_resnet101bt.pkl')

    name_list,res50bt_feature=load_feature(res50bt_feature_path)
    # name_list,res101bt_feature=load_feature(res101bt_feature_path)

    # with open("./feature/feat_resnet50bt_mac.pkl", "wb") as file_to_save:
    #     pickle.dump(
    #         {
    #         'name':name_list,
    #         'Mac':res50bt_feature['Mac']
    #             }, 
    #         file_to_save, 
    #         -1
    #         )

    feature={
        'resnet50bt':res50bt_feature,
        # 'resnet101bt':res101bt_feature
    }

    feat_type={
        'resnet50bt':['Mac'],
        # 'resnet101bt':['Mac']
    }
    weight={
        'resnet50bt':{'Mac':1,'rmac':0,'ramac':0,'Grmac':0},
        # 'resnet101bt':{'Mac':1,'rmac':0,'ramac':0,'Grmac':0},
    }
    dim_feature={
        'resnet50bt':2048,
        # 'resnet101bt':2048,
    }
    batch_size=20

    similarity=torch.zeros(len(os.listdir(test_image_path)),len(name_list))
    print(similarity.size())

    for model_name in ['resnet50bt']:
        feature_model=feature[model_name]
        for item in feat_type[model_name]:
            if model_name == 'resnet50bt':
                model=resnet50bt(res50bt_path,item)
            elif model_name == 'resnet101bt':
                model=resnet101bt(res101bt_path,item)
            else:
                pass

            feat_reserved=feature_model[item]

            dataset = retrieval_dataset(test_image_path,transform=transform_480)
            testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            model=model.cpu()
            model.eval()
            query=torch.empty(len(os.listdir(test_image_path)),dim_feature[model_name])
            name_test=[]
            with torch.no_grad():
                for i, (inputs, names) in tqdm(enumerate(testloader)):
                    query[i*batch_size:i*batch_size+len(names)] = model(inputs).cpu()
                    name_test.extend(names)

            feat_reserved=torch.Tensor(feat_reserved).transpose(1,0)
            query=torch.Tensor(query)
            
            similarity+=torch.matmul(query,feat_reserved)*weight[model_name][item]

    _, predicted = similarity.topk(7)
    predicted=predicted.tolist()
    dict_result=dict(zip(name_test,predicted))

    #saving csv
    img_results=[]
    name_test.sort()
    for name in name_test:
        temp=[name.split('.')[0]]
        for idx in dict_result[name]:
            temp.append(name_list[idx].split('.')[0])
        img_results.append(temp)
    print('saving')
    out = open(result_path,'w', encoding='utf-8', newline='')
    csv_write = csv.writer(out)
    csv_write.writerows(img_results)
