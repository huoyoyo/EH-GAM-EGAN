import torch


def generate_fc_edge_index(num_nodes):
    # 生成节点索引
    node_index = torch.arange(num_nodes)

    # 生成所有可能的边索引
    edge_index = torch.cartesian_prod(node_index, node_index)

    # 过滤掉自环边索引和重复边索引
    edge_index = edge_index[edge_index[:, 0] != edge_index[:, 1]]

    # 将边索引按照形状[2, num_edges]返回
    fc_edge_index = edge_index.t()

    # 交换行
    swapped_tensor = fc_edge_index[[1, 0], :]

    return swapped_tensor

if __name__ == '__main__':
    print(generate_fc_edge_index(6))
