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
    parser.add_argument("--delta", type=float, default=0.2)
    
    opt = parser.parse_args()
    return opt

def generate_desire_distribution(target_distribution, total_label, desire_delta=0.2):
    desire_distribution = [0 for _ in range(total_label)]
    random_value = list(range(total_label * 10))

    total_loop = 1000
    for loop in range(total_loop):
        for i in range(total_label):
            desire_distribution[i] = random.choice(random_value)

        desire_distribution = np.array(desire_distribution)
        target_distribution = np.array(target_distribution)

        desire_distribution = desire_distribution / desire_distribution.sum()

        delta = ((desire_distribution - target_distribution) ** 2).sum()
        delta = np.sqrt(delta)
        
        if delta >= desire_delta:
            return desire_distribution
    
    raise Exception("Cannot generate distribution with desire delta")

def partition(desire_distribution, labels, num_clients, minvol=6, num_classes=10, skewness=0.1, sample_per_label=1000):
    # sample data for desire_distribution
    idxs = range(len(labels))
    idxs_labels = np.vstack((idxs, labels))
    print('Idxs_labels: ', idxs_labels.shape)
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    print('Idxs: ', idxs)
    labels = idxs_labels[1, :]
    print('Labels: ', labels)
    tmp = 0
    list_tmp = []

    for i in range(len(labels)):
        if(labels[i] == tmp):
            tmp +=1
            list_tmp.append(i)
    list_tmp.append(len(labels))

    train_data = []
    maxx = max(desire_distribution)
    for i in range(num_classes):
        num_sample = int(sample_per_label * desire_distribution[i] / maxx)
        label_data = random.sample(list(range(list_tmp[i], list_tmp[i+1])), k=num_sample)
        for data in label_data:
            train_data.append(idxs[data])

    # labels skew pareto
    min_size = 0
    dpairs = [[did, labels[did]] for did in train_data]
    # dpairs = [[did, self.train_data[did][-1]]
                # for did in range(len(self.train_data))]
    local_datas = [[] for _ in range(num_clients)]
    while min_size < minvol:
        idx_batch = [[] for i in range(num_clients)]
        for k in range(num_classes):
            idx_k = [p[0] for p in dpairs if p[1] == k]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(
                np.repeat(skewness, num_clients))
            # Balance
            proportions = np.array(
                [p * (len(idx_j) < len(labels) / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) *
                            len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,
                            idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    for j in range(num_clients):
        np.random.shuffle(idx_batch[j])
        local_datas[j].extend(idx_batch[j])

    return local_datas

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
    
    desire_distribution = generate_desire_distribution(target_distribution=opt.target_distribution,
                                                       total_label=opt.total_label,
                                                       desire_delta=opt.delta)
    local_datas = partition(desire_distribution=desire_distribution, labels=labels,
                            num_clients=opt.num_client, minvol=opt.minvol,
                            num_classes=opt.total_label, skewness=opt.skew,
                            sample_per_label=opt.sample_per_label)
    
    # local_datas = partition(labels, total_label=opt.total_label, total_client=opt.num_client,
    #                              sample_per_label=opt.sample_per_label, class_per_client=opt.client_class,
    #                              min_sample=opt.minvol)
    
    fed_data = local_holdout(local_datas)
    
    task = f"{option.dataset}_cnum{option.num_client}_dist11_skew{option.skew}_seed0"
    
    
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
    info_data["dist"] = 11
    info_data["skewness"] = option.skew
    info_data["num-clients"] = option.num_client
    
    json.dump(info_data, open(info_path, "w"))
    
    
    
    

if __name__ == "__main__":
    opt = parse_opt()
    
    if opt.dataset == "cifar10":
        opt.total_label = 10
        opt.sample_per_label = 5000
        opt.target_distribution = [0.1 for i in range(10)]
        
        rawdata_path = "./benchmark/cifar10/data"
        
        train_data = datasets.CIFAR10(rawdata_path, train=True, download=True)
        test_data = datasets.CIFAR10(rawdata_path, train=False, download=True)
                
    elif opt.dataset == "cifar100":
        opt.total_label = 100
        opt.sample_per_label = 500
        opt.target_distribution = [0.01 for i in range(100)]
        
        rawdata_path = "./benchmark/cifar100/data"
        
        train_data = datasets.CIFAR100(rawdata_path, train=True, download=True)
        test_data = datasets.CIFAR100(rawdata_path, train=False, download=True)
        
    generate_fedtask(opt, train_data, test_data)
    
    