from common_utils import *
from params import configs
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
from tqdm import tqdm
from data_utils import pack_data_from_config
import time
import numpy as np
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id

def test_heuristic_method(data_set, method, seed):
    """
    test one heuristic method on the given data, run 100 times.

    :param data_set:  test data
    :param method:    the name of heuristic method (e.g. 'FIFO-FIFO')
    :param seed:      seed for testing
    :return:          np.array of shape (100, 2), each row is [makespan, time]
    """
    setup_seed(seed)
    result = []

    # 这里固定跑 100 次
    for i in tqdm(range(len(data_set[0])), file=sys.stdout, desc="progress", colour='blue'):
        # 只取 data_set[0][0] 里的数据做测试，如果后续要对多个实例循环，可以再改
        n_j = data_set[0][i].shape[0]
        n_op, n_m = data_set[1][i].shape

        env = FJSPEnvForSameOpNums(n_j=n_j, n_m=n_m)
        env.set_initial_data([data_set[0][4]], [data_set[1][4]])

        t1 = time.time()
        while True:
            action = combined_heuristic_select_action(method, env, e=0)
            _, _, done = env.step(np.array([action]))
            if done:
                break
        t2 = time.time()

        result.append([env.current_makespan[0], t2 - t1])

    return np.array(result)


def main():
    """
    test heuristic methods following the config and save the results:
    each method runs 100 times on the first instance in data_set
    """
    setup_seed(configs.seed_test)
    if not os.path.exists('./test_results'):
        os.makedirs('./test_results')

    # 打包测试数据
    test_data = pack_data_from_config(configs.data_source, configs.test_data)

    # 如果 test_method 为空，就默认 8 种组合
    if len(configs.test_method) == 0:
        test_method = [
            'FIFO-FIFO', 'FIFO-SWT',
            'MOR-FIFO',  'MOR-SWT',
            'SPT-FIFO',  'SPT-SWT',
            'MWKR-FIFO', 'MWKR-SWT'
        ]
    else:
        test_method = configs.test_method

    # 遍历测试集
    for data in test_data:
        print("-" * 25 + "Test Heuristic Methods" + "-" * 25)
        print('Test Methods:', test_method)
        print(f"test data name: {configs.data_source},{data[1]}")

        save_direc = f'./test_results/{configs.data_source}/{data[1]}'
        if not os.path.exists(save_direc):
            os.makedirs(save_direc)

        # 针对每种方法，跑 100 次，保存结果
        for method in test_method:
            save_path = os.path.join(save_direc, f'Result_{method}_{data[1]}.npy')

            # 如果结果文件不存在，或者要覆盖
            if (not os.path.exists(save_path)) or configs.cover_heu_flag:
                print(f"Heuristic method : {method}")
                seed = configs.seed_test

                # 只跑一次 test_heuristic_method，就做 100 次测试
                result = test_heuristic_method(data[0], method, seed)
                # result shape: (100, 2)

                # 输出一下平均结果，方便查看
                print(f"makespan mean: {np.mean(result[:, 0]):.2f}, "
                      f"time mean: {np.mean(result[:, 1]):.4f}")

                # 保存所有 100 次结果
                np.save(save_path, result)

if __name__ == '__main__':
    main()
