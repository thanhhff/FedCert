# python generate_fedtask.py --dataset mnist --dist 0 --skew 0 --num_clients 100
# python generate_fedtask.py --dataset mnist --dist 1 --skew 0.8 --num_clients 10
# python generate_fedtask.py --dataset mnist --dist 2 --skew 0.8 --num_clients 10
# python generate_fedtask.py --dataset mnist --dist 3 --skew 0.8 --num_clients 10

# python generate_fedtask.py --dataset cifar100 --dist 0 --skew 0 --num_clients 100
# python generate_fedtask.py --dataset cifar100 --dist 2 --skew 0.5 --num_clients 100
# python generate_fedtask.py --dataset cifar100 --dist 6 --skew 0.5 --num_clients 100

# python generate_fedtask.py --dataset cifar100 --dist 2 --skew 0.2 --num_clients 100
# python generate_fedtask.py --dataset cifar100 --dist 2 --skew 0.7 --num_clients 100

# # 7-4-2023
# python generate_fedtask.py --dataset cifar10 --dist 2 --skew 0.5 --num_clients 100
# python generate_fedtask.py --dataset cifar10 --dist 6 --skew 0.5 --num_clients 100
# python generate_fedtask.py --dataset cifar10 --dist 0 --skew 0 --num_clients 100

# 10-4-2023
# python generate_fedtask.py --dataset cifar10 --dist 3 --skew 0.5 --num_clients 100
# python generate_fedtask.py --dataset cifar10 --dist 2 --skew 0.1 --num_clients 100

# 13-4-2023
# python generate_fedtask.py --dataset cifar10 --dist 7 --num_clients 100
# python generate_fedtask_aLong.py --dataset cifar10 --dist 8 --skew 0.5 --num_clients 100

# 21-4-2023
# python generate_fedtask.py --dataset cifar10 --dist 9 --skew 0.5 --num_clients 100

# 28-4-2023
# python generate_fedtask.py --dataset cifar10 --dist 10 --skew 0.1 --num_clients 100
# python generate_fedtask.py --dataset cifar10 --dist 10 --skew 0.3 --num_clients 100
# python generate_fedtask.py --dataset cifar10 --dist 10 --skew 0.5 --num_clients 100

# 2-5-2023
# python generate_fedtask.py --dataset cifar10 --dist -1 --skew 0.1 --num_clients 100
# python generate_fedtask.py --dataset cifar10 --dist 10 --skew 0.5 --num_clients 100
# python generate_fedtask.py --dataset cifar10 --dist 10 --skew 0.3 --num_clients 100
# python generate_fedtask.py --dataset cifar10 --dist 10 --skew 0.1 --num_clients 100


# 4-5-2023
# python generate_fedtask.py --dataset cifar10 --dist 8 --skew 0.5 --num_clients 100
# python generate_fedtask_aLong.py --dataset cifar10 --dist 8 --skew 0.5 --num_clients 100
# python generate_fedtask_aLong.py --dataset cifar10 --dist 8 --skew 0.1 --num_clients 100

# python noniid_cifar100.py --method featured --total_client 50
# python noniid_cifar100.py --method quantitative --total_client 50
# python noniid_cifar100.py --method pareto --total_client 50
# python noniid_cifar100.py --method cluster --total_client 50

# 25-5-2023
python generate_fedtask_aLong.py --dataset cifar100 --dist true_pareto --skew 0.5 --num_clients 50 --file_path /mnt/disk1/naver/hieunguyen/provably_fl/gendata/gendata/cifar/cifar100/50client/CIFAR100_50client_pareto.json
python generate_fedtask_aLong.py --dataset cifar100 --dist true_cluster --skew 0.5 --num_clients 50 --file_path /mnt/disk1/naver/hieunguyen/provably_fl/gendata/gendata/cifar/cifar100/50client/CIFAR100_50client_cluster.json
python generate_fedtask_aLong.py --dataset cifar100 --dist true_featured --skew 0.5 --num_clients 50 --file_path /mnt/disk1/naver/hieunguyen/provably_fl/gendata/gendata/cifar/cifar100/50client/CIFAR100_50client_featured.json
python generate_fedtask_aLong.py --dataset cifar100 --dist true_quantitative --skew 0.5 --num_clients 50 --file_path /mnt/disk1/naver/hieunguyen/provably_fl/gendata/gendata/cifar/cifar100/50client/CIFAR100_50client_quantitative.json


