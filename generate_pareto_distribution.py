import argparse
import numpy as np
import random
from torchvision import datasets
import math
import json
import os

def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--skew", type=float, default=0.1)
    parser.add_argument("--minvol", type=int, default=12)
    parser.add_argument("--client_class", type=int, default=3)
    parser.add_argument("--num_client", type=int, default=100)
    
    opt = parser.parse_args()
    return opt

def partition_data(labels, total_label, total_client, sample_per_label, class_per_client, min_sample):
    # total_client = 100
    # total_label = 10
    # sample_per_label = 1000
    # class_per_client = 3
    # min_sample = 12

    idxs = range(len(labels))
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]
    tmp = 0
    list_tmp = []

    for i in range(len(labels)):
        if(labels[i] == tmp):
            tmp +=1
            list_tmp.append(i)
    list_tmp.append(len(labels))

    while(True):
        list_label ={}
        a = set()
        for i in range(total_client):
            list_label[i] = random.sample(range(total_label), class_per_client)
            a.update(list_label[i])
        if len(a) >= total_label:
            break

    key = True
    count = 0
    decrease_sample = 20

    while(key):
        count += 1
        if count > 200:
            decrease_sample += 1
            print("Infinite loop")
            print(count)
#             break

        try:
            list_dict = [0] * total_label
            for i in range(total_label):
                list_dict[i] = idxs[list_tmp[i]:list_tmp[i+1]]

            dis = np.random.pareto(tuple([1] * total_client))
            dis = dis/np.sum(dis)
            percent = [0] * total_label
            for i in range(total_client):
                for j in list_label[i]:
                    percent[j] += dis[i] / class_per_client

            maxx = max(percent)
            total = (sample_per_label - decrease_sample) / maxx
            
            sample_client = [math.ceil(total * dis[i]) for i in range(total_client)]
            for i in range(len(sample_client)):
                if sample_client[i] < min_sample:
                    sample_client[i] = min_sample

            dict_client = [[] for i in range(total_client)]
            for i in range(total_client):
                dict_client[i] = []

            for i in range(total_client):

                x = math.ceil(sample_client[i] / class_per_client)
                for j in list_label[i]:
                    a = np.random.choice(list_dict[j], x, replace=False)
                    list_dict[j] = list(set(list_dict[j]) - set(a))
                    dict_client[i] = dict_client[i] + list(a)


                dict_client[i] = [int(j) for j in dict_client[i]]
            key = False
        except ValueError:
            key = True
            
    return dict_client

def local_holdout(local_datas, rate=0.8):
    fed_data = []
    for client_data in local_datas:
        random.shuffle(client_data)
        train_size = int(len(client_data) * rate)
        train_data = client_data[:train_size]
        val_data = client_data[train_size:]
        fed_data.append({
            'dtrain': train_data,
            'dvalid': val_data
        })
    return fed_data

def get_client_names(client_id):
    if client_id < 10:
        return f"00{client_id}"
    if client_id < 100:
        return f"0{client_id}"
    

def generate_fedtask(option: argparse.Namespace, train_data, test_data):
    labels = train_data.targets
    local_datas = partition_data(labels, total_label=opt.total_label, total_client=opt.num_client,
                                 sample_per_label=opt.sample_per_label, class_per_client=opt.client_class,
                                 min_sample=opt.minvol)
    
    fed_data = local_holdout(local_datas)
    
    task = f"{option.dataset}_cnum{option.num_client}_dist10_skew{option.skew}_seed0"
    
    
    fedtask_path = os.path.join("./fedtask", task)
    if not os.path.exists(fedtask_path):
        os.makedirs(fedtask_path)
        os.makedirs(os.path.join(fedtask_path, "record"))
    else:
        assert False, "Task have already exists"
    
    
    # saving fed task
    data_path = os.path.join(fedtask_path, "data.json")
    info_path = os.path.join(fedtask_path, "info.json")
    
    saving_data = {}
    
    saving_data["store"] = "XY"
    saving_data["client_names"] = []
    saving_data["dtest"] = list(range(len(test_data)))
    
    # for client in saving_data["client_names"]:
    #     name = get_client_names(client)
    #     saving_data[name] = fed_data[client]
    
    for i, client_data in enumerate(fed_data):
        name = get_client_names(i)
        saving_data["client_names"].append(name)
        saving_data[name] = client_data
        
    json.dump(saving_data, open(data_path, "w"))
    
    info_data = {}
    info_data["benchmark"] = option.dataset
    info_data["dist"] = 10
    info_data["skewness"] = option.skew
    info_data["num-clients"] = option.num_client
    
    json.dump(info_data, open(info_path, "w"))
    
    
    
    

if __name__ == "__main__":
    opt = parse_opt()
    
    if opt.dataset == "cifar10":
        opt.total_label = 10
        opt.sample_per_label = 5000
        
        rawdata_path = "./benchmark/cifar10/data"
        
        train_data = datasets.CIFAR10(rawdata_path, train=True, download=True)
        test_data = datasets.CIFAR10(rawdata_path, train=False, download=True)
                
    elif opt.dataset == "cifar100":
        opt.total_label = 100
        opt.sample_per_label = 500
        
        rawdata_path = "./benchmark/cifar100/data"
        
        train_data = datasets.CIFAR100(rawdata_path, train=True, download=True)
        test_data = datasets.CIFAR100(rawdata_path, train=False, download=True)
        
    generate_fedtask(opt, train_data, test_data)
    
    
