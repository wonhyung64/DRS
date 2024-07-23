import torch
import numpy as np


def em_clustering(
    batch_size,
    env_num,
    num_sample,
    total_batch,
    device,
    model,
    x_train,
    y_train,
    envs,
    const_env_tensor_list,
    cluster_distance_func,
    ) -> int:

    model.eval()
    new_env_tensors_list: list = []

    all_idx = np.arange(num_sample)
    for idx in range(total_batch+1):
        selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
        sub_x = x_train[selected_idx]
        sub_x = torch.LongTensor(sub_x - 1).to(device)
        batch_users_tensor = sub_x[:,0]
        batch_items_tensor = sub_x[:,1]
        sub_y = y_train[selected_idx]
        sub_y = torch.Tensor(sub_y).to(device)

        distances_list: list = []
        for env_idx in range(env_num):
            envs_tensor = const_env_tensor_list[env_idx][0:batch_users_tensor.shape[0]]
            cluster_pred = model.cluster_predict(batch_users_tensor, batch_items_tensor, envs_tensor)
            distances = cluster_distance_func(cluster_pred, sub_y)
            distances = distances.reshape(-1, 1)
            distances_list.append(distances)
        each_envs_distances = torch.cat(distances_list, dim=1)

        new_envs = torch.argmin(each_envs_distances, dim=1)
        new_env_tensors_list.append(new_envs)

    all_new_env_tensors = torch.cat(new_env_tensors_list, dim=0)
    envs_diff = (envs - all_new_env_tensors) != 0
    diff_num = int(torch.sum(envs_diff))

    return all_new_env_tensors, diff_num
