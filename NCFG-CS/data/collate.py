from typing import List, Tuple, Optional, Union, Any

from collections import defaultdict
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor
from torch_sparse import SparseTensor, cat

from torch_geometric.data.data import BaseData
from torch_geometric.data.storage import BaseStorage, NodeStorage

mini_inc = []                                           # 用来额外记录子图的x的增量关系

def collate(
    cls,
    data_list: List[BaseData],
    increment: bool = True,
    add_batch: bool = True,
    follow_batch: object = None,
    exclude_keys: object = None,
    # follow_batch: Optional[Union[List[str]]] = None,
    # exclude_keys: Optional[Union[List[str]]] = None,
) -> Tuple[BaseData, Mapping, Mapping]:
    # Collates a list of `data` objects into a single object of type `cls`.
    # `collate` can handle both homogeneous and heterogeneous data objects by
    # individually collating all their stores.
    # In addition, `collate` can handle nested data structures such as
    # dictionaries and lists.

    if not isinstance(data_list, (list, tuple)):
        # Materialize `data_list` to keep the `_parent` weakref alive.
        data_list = list(data_list)

    if cls != data_list[0].__class__:
        out = cls(_base_cls=data_list[0].__class__)  # Dynamic inheritance.
    else:
        out = cls()

    # Create empty stores:
    out.stores_as(data_list[0])

    follow_batch = set(follow_batch or [])
    exclude_keys = set(exclude_keys or [])

    # Group all storage objects of every data object in the `data_list` by key,
    # i.e. `key_to_store_list = { key: [store_1, store_2, ...], ... }`:
    key_to_stores = defaultdict(list)
    for data in data_list:
        for store in data.stores:
            key_to_stores[store._key].append(store)

    # With this, we iterate over each list of storage objects and recursively
    # collate all its attributes into a unified representation:

    # We maintain two additional dictionaries:
    # * `slice_dict` stores a compressed index representation of each attribute
    #    and is needed to re-construct individual elements from mini-batches.
    # * `inc_dict` stores how individual elements need to be incremented, e.g.,
    #   `edge_index` is incremented by the cumulated sum of previous elements.
    #   We also need to make use of `inc_dict` when re-constructuing individual
    #   elements as attributes that got incremented need to be decremented
    #   while separating to obtain original values.
    device = None
    # 维护两个字典 -- 分别表示前面累加的向量以及数据增量的向量
    slice_dict, inc_dict = defaultdict(dict), defaultdict(dict)
    for out_store in out.stores:
        key = out_store._key
        stores = key_to_stores[key]
        for attr in stores[0].keys():

            if attr in exclude_keys:  # Do not include top-level attribute.
                continue
            # 这里的attr会根据传入的key值,拼接成一个统一的向量 -- attr 有x, edge_index等等；x此时是sum * 50
            values = [store[attr] for store in stores]

            # The `num_nodes` attribute needs special treatment, as we need to
            # sum their values up instead of merging them to a list:
            if attr == 'num_nodes':
                out_store._num_nodes = values
                out_store.num_nodes = sum(values)
                continue

            # Skip batching of `ptr` vectors for now:
            if attr == 'ptr':
                continue

            # Collate attributes into a unified representation:   -- 此时向量处理完毕 -- 此时的edge_index就已经评完
            value, slices, incs = _collate(attr, values, data_list, stores,
                                           increment)

            # 对mini_x_batch作额外处理 -- 分为一个mini_batch处理数据源
            # if attr == 'mini_x_batch':
            #     unique_values, inverse_indices = torch.unique(value, return_inverse=True)
            #     value = inverse_indices
            device = value.device if isinstance(value, Tensor) else device

            out_store[attr] = value                                     
            if key is not None:
                slice_dict[key][attr] = slices
                inc_dict[key][attr] = incs
            else:
                slice_dict[attr] = slices
                inc_dict[attr] = incs

            # Add an additional batch vector for the given attribute:
            if (attr in follow_batch and isinstance(slices, Tensor)
                    and slices.dim() == 1):
                repeats = slices[1:] - slices[:-1]
                arange = torch.arange(len(values), device=device)
                batch = arange.repeat_interleave(repeats.to(device))
                out_store[f'{attr}_batch'] = batch

        # In case the storage holds node, we add a top-level batch vector it:
        if (add_batch and isinstance(stores[0], NodeStorage)
                and stores[0].can_infer_num_nodes):
            arange = torch.arange(len(stores), device=device)
            repeats = torch.tensor([store.num_nodes for store in stores],
                                   device=device)
            out_store.batch = arange.repeat_interleave(repeats)
            out_store.ptr = cumsum(repeats)

    return out, slice_dict, inc_dict

# 做具体的向量拼接的工作
def _collate(
    key: str,
    values: List[Any],
    data_list: List[BaseData],
    stores: List[BaseStorage],
    increment: bool,
) -> Tuple[Any, Any, Any]:

    elem = values[0]
    if isinstance(elem, Mapping):
        # Recursively collate elements of dictionaries.
        value_dict, slice_dict, inc_dict = {}, {}, {}
        for key in elem.keys():
            value_dict[key], slice_dict[key], inc_dict[key] = _collate(
                key, [v[key] for v in values], data_list, stores, increment)
        return value_dict, slice_dict, inc_dict

    elif (isinstance(elem, Sequence) and not isinstance(elem, str)
          and isinstance(elem[0], (Tensor, SparseTensor))):
        # Recursively collate elements of lists.
        value_list, slice_list, inc_list = [], [], []
        for i in range(len(elem)):
            value, slices, incs = _collate(key, [v[i] for v in values],
                                           data_list, stores, increment)
            value_list.append(value)
            slice_list.append(slices)
            inc_list.append(incs)
        return value_list, slice_list, inc_list

    elif isinstance(elem, Tensor):
        # Concatenate a list of `torch.Tensor` along the `cat_dim`.
        # NOTE: We need to take care of incrementing elements appropriately.
        cat_dim = data_list[0].__cat_dim__(key, elem, stores[0])
        # 增加自己的需求 -- 其余的默认为0 -- mini_edge默认为 -1
        if key == "mini_edge":
            cat_dim = -1
        if cat_dim is None or elem.dim() == 0:
            values = [value.unsqueeze(0) for value in values]
        slices = cumsum([value.size(cat_dim or 0) for value in values])         # 返回此前累计的向量,从0开始,[0,3,5]表示后面的两个Tensor的x的行数分别为3, 2；返回edge_index边的累计集合
        if increment:      
            incs = get_incs(key, values, data_list, stores)
            if incs.dim() > 1 or int(incs[-1]) != 0:
                values = [value + inc for value, inc in zip(values, incs)]      # 实现增量的相加，这个是根据x -- 即节点的累积量决定的
        else:
            incs = None

        value = torch.cat(values, dim=cat_dim or 0)                             # 可能将index拼接好
        return value, slices, incs                                              # 返回的value是拼接好的向量,如x[50, 50]; slices是各自行数的叠加[0, 3, 5],inc全0

    elif isinstance(elem, SparseTensor) and increment:
        # Concatenate a list of `SparseTensor` along the `cat_dim`.
        # NOTE: `cat_dim` may return a tuple to allow for diagonal stacking.
        cat_dim = data_list[0].__cat_dim__(key, elem, stores[0])
        cat_dims = (cat_dim, ) if isinstance(cat_dim, int) else cat_dim
        repeats = [[value.size(dim) for dim in cat_dims] for value in values]
        slices = cumsum(repeats)
        value = cat(values, dim=cat_dim)
        return value, slices, None

    elif isinstance(elem, (int, float)):
        # Convert a list of numerical values to a `torch.Tensor`.
        value = torch.tensor(values)
        if increment:
            incs = get_incs(key, values, data_list, stores)
            if int(incs[-1]) != 0:
                value.add_(incs)
        else:
            incs = None
        slices = torch.arange(len(values) + 1)
        return value, slices, incs

    else:
        # Other-wise, just return the list of values as it is.
        slices = torch.arange(len(values) + 1)
        return values, slices, None


###############################################################################

# 返回累计组成的向量对应的各自位置 [0, 3, 5, 15, 18……]
def cumsum(value: Union[Tensor, List[int]]) -> Tensor:
    if not isinstance(value, Tensor):
        value = torch.tensor(value)
    out = value.new_empty((value.size(0) + 1, ) + value.size()[1:])
    out[0] = 0
    torch.cumsum(value, 0, out=out[1:])
    return out

# 这段代码的功能是计算一组数据对象列表 data_list 中的每个数据对象的 key 属性增量值的累积，并返回结果
def get_incs(key, values: List[Any], data_list: List[BaseData],
             stores: List[BaseStorage]) -> Tensor:
    if key == 'mini_edge':
        repeats = [temData.mini_x.shape[0] for temData in data_list]
    else:
        repeats = [
        data.__inc__(key, value, store)
        for value, data, store in zip(values, data_list, stores)
    ] # 返回一个batch中各个x有多少行[3, 2, 10, 3], 即有多少个节点
    if isinstance(repeats[0], Tensor):
        repeats = torch.stack(repeats, dim=0)
    else:
        repeats = torch.tensor(repeats)
    return cumsum(repeats[:-1])
