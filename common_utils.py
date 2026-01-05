import json
import random

from torch.distributions.categorical import Categorical
import sys
import numpy as np
import torch
import copy

"""
    agent utils
"""


def sample_action(p):
    """
        sample an action by the distribution p
    :param p: this distribution with the probability of choosing each action
    :return: an action sampled by p
    """
    dist = Categorical(p)
    s = dist.sample()  # index
    return s, dist.log_prob(s)


def eval_actions(p, actions):
    """
    :param p: the policy
    :param actions: action sequences
    :return: the log probability of actions and the entropy of p
    """
    softmax_dist = Categorical(p.squeeze())
    ret = softmax_dist.log_prob(actions).reshape(-1)
    entropy = softmax_dist.entropy().mean()
    return ret, entropy


def greedy_select_action(p):
    _, index = torch.max(p, dim=-1)
    return index


def min_element_index(array):
    """
    :param array: an array with numbers
    :return: Index set corresponding to the minimum element of the array
    """
    min_element = np.min(array)
    candidate = np.where(array == min_element)
    return candidate


def max_element_index(array):
    """
    :param array: an array with numbers
    :return: Index set corresponding to the maximum element of the array
    """
    max_element = np.max(array)
    candidate = np.where(array == max_element)
    return candidate


def available_mch_list_for_job(chosen_job, env):
    """
    :param chosen_job: the selected job
    :param env: the production environment
    :return: the machines which can immediately process the chosen job
    """
    mch_state = ~env.candidate_process_relation[0, chosen_job]
    available_mch_list = np.where(mch_state == True)[0]
    mch_free_time = env.mch_free_time[0][available_mch_list]
    job_free_time = env.candidate_free_time[0][chosen_job]
    # case1 eg:
    # JF: 50
    # MchF: 55 60 65 70
    if (job_free_time < mch_free_time).all():
        chosen_mch_list = available_mch_list[min_element_index(mch_free_time)]
    # case2 eg:
    # JF: 50
    # MchF: 35 40 55 60
    else:
        chosen_mch_list = available_mch_list[np.where(mch_free_time <= job_free_time)]

    return chosen_mch_list


def heuristic_select_action(method, env):
    """
    :param method: the name of heuristic method
    :param env: the environment
    :return: the action selected by the heuristic method

    here are heuristic methods selected for comparison:

    FIFO: First in first out
    MOR(or MOPNR): Most operations remaining
    SPT: Shortest processing time
    MWKR: Most work remaining
    """
    chosen_job = -1
    chosen_mch = -1

    job_state = (env.mask[0] == 0)

    process_job_state = (env.candidate_free_time[0] <= env.next_schedule_time[0])
    job_state = process_job_state & job_state

    available_jobs = np.where(job_state == True)[0]
    available_ops = env.candidate[0][available_jobs]

    if method == 'FIFO':
        # selecting the earliest ready candidate operation
        candidate_free_time = env.candidate_free_time[0][available_jobs]
        chosen_job_list = available_jobs[min_element_index(candidate_free_time)]
        chosen_job = np.random.choice(chosen_job_list)

        # select the earliest ready machine
        mch_state = ~env.candidate_process_relation[0, chosen_job]
        available_mchs = np.where(mch_state == True)[0]
        mch_free_time = env.mch_free_time[0][available_mchs]
        chosen_mch_list = available_mchs[min_element_index(mch_free_time)]
        chosen_mch = np.random.choice(chosen_mch_list)

    elif method == 'MOR':
        remain_ops = env.op_match_job_left_op_nums[0][available_ops]
        chosen_job_list = available_jobs[max_element_index(remain_ops)]
        chosen_job = np.random.choice(chosen_job_list)

        # select a machine which can immediately process the chosen job
        chosen_mch_list = available_mch_list_for_job(chosen_job, env)
        chosen_mch = np.random.choice(chosen_mch_list)

    elif method == 'SPT':

        temp_pt = copy.deepcopy(env.candidate_pt[0])
        temp_pt[env.dynamic_pair_mask[0]] = float("inf")
        pt_list = temp_pt.reshape(-1)

        action_list = np.where(pt_list == np.min(pt_list))[0]

        action = np.random.choice(action_list)
        return action

    elif method == 'MWKR':
        job_remain_work_list = env.op_match_job_remain_work[0][available_ops]

        chosen_job = available_jobs[np.random.choice(max_element_index(job_remain_work_list)[0])]

        # select a machine which can immediately process the chosen job
        chosen_mch_list = available_mch_list_for_job(chosen_job, env)
        chosen_mch = np.random.choice(chosen_mch_list)

    else:
        print(f'Error From rule select: undefined method {method}')
        sys.exit()

    if chosen_job == -1 or chosen_mch == -1:
        print(f'Error From choosing action: choose job {chosen_job}, mch {chosen_mch}')
        sys.exit()

    action = chosen_job * env.number_of_machines + chosen_mch
    return action


def heuristic_select_action_for_job(method, env, e):
    chosen_job = -1

    job_state = (env.mask[e] == 0)

    process_job_state = (env.candidate_free_time[e] <= env.next_schedule_time[e])
    job_state = process_job_state & job_state

    available_jobs = np.where(job_state == True)[0]
    available_ops = env.candidate[e][available_jobs]

    if method == 'FIFO':
        # selecting the earliest ready candidate operation (FIFO rule)
        candidate_free_time = env.candidate_free_time[e][available_jobs]
        chosen_job_list = available_jobs[min_element_index(candidate_free_time)]
        if len(chosen_job_list) == 0:
            return -1
        chosen_job = np.random.choice(chosen_job_list)

    elif method == 'MOR':
        remain_ops = env.op_match_job_left_op_nums[e][available_ops]
        chosen_job_list = available_jobs[max_element_index(remain_ops)]
        if len(chosen_job_list) == 0:
            return -1
        chosen_job = np.random.choice(chosen_job_list)

    elif method == 'SPT':
        temp_pt = copy.deepcopy(env.candidate_pt[e])
        temp_pt[env.dynamic_pair_mask[e]] = float("inf")
        pt_list = temp_pt.reshape(-1)

        job_list = np.where(pt_list == np.min(pt_list))[0]

        if len(job_list) == 0:
            return -1

        chosen_job = np.random.choice(job_list)
        chosen_job = chosen_job // env.number_of_machines
        # print('J ', chosen_job)

    elif method == 'MWKR':
        job_remain_work_list = env.op_match_job_remain_work[e][available_ops]
        # print(env.op_match_job_remain_work[e][available_ops])
        chosen_job = available_jobs[np.random.choice(max_element_index(job_remain_work_list)[0])]

    return chosen_job


def select_machine(env, e, rule_mch, selected_op):
    """
    Select machine based on the specified PDR rule, considering dynamic_pair_mask for the selected operation.

    :param e: Environment index
    :param rule_mch: Machine PDR rule index
    :param selected_op: Previously selected operation
    :return: selected_mch or None if no valid machine
    """
    selected_mch = -1
    M = env.number_of_machines

    # Convert flat operation index to (j, m)
    j = selected_op // M

    if rule_mch == 0:  # FIFO
        # 找到空闲时间最长的机器
        # print(selected_op)
        mch_state = ~env.candidate_process_relation[e, selected_op]
        available_mchs = np.where(mch_state == True)[0]
        if len(available_mchs) == 0:
            return None  # 无可用机器
        # available_mchs = np.where(mch_state == True)[0]
        mch_free_time = env.mch_free_time[e][available_mchs]
        chosen_mch_list = available_mchs[min_element_index(mch_free_time)]
        selected_mch = np.random.choice(chosen_mch_list)

    elif rule_mch == 1:  # MWT
        # 找到可处理选定操作的机器
        mch_state = ~env.candidate_process_relation[e, selected_op]
        available_mchs = np.where(mch_state == True)[0]

        # 获取可用机器的等待时间
        mch_wait_time = env.mch_waiting_time[e][available_mchs]
        # 找到最小等待时间
        min_wait_time = np.max(mch_wait_time)
        # 找到所有等待时间等于最大值的机器
        chosen_mch_list = available_mchs[mch_wait_time == min_wait_time]
        # 随机选择一台
        selected_mch = np.random.choice(chosen_mch_list)

    return selected_mch


def combined_heuristic_select_action(method, env, e=0):
    """
    根据组合的启发式方法 (job_rule - mch_rule)，先选择作业，再选择机器，最后返回动作索引 action。

    :param method: 形如 'FIFO-FIFO', 'FIFO-SWT', 'MOR-FIFO', 'MOR-SWT',
                   'SPT-FIFO', 'SPT-SWT', 'MWKR-FIFO', 'MWKR-SWT' 等
    :param env: FJSP 环境
    :param e:   environment index，一般是 0（若是并行环境或多环境，可以根据需要调整）
    :return: action (int) = job_id * env.number_of_machines + machine_id
    """
    # 1) 解析组合字符串或元组
    # 假设是字符串 "SPT-SWT"
    try:
        job_rule, mch_rule = method.split('-')
    except ValueError:
        # 如果不是用 '-' 分隔，可根据你的代码约定修改解析方式
        print(f'[Error] Invalid method format: {method}')
        sys.exit(1)

    # 2) 根据 job_rule 选作业
    selected_job = heuristic_select_action_for_job(job_rule, env, e)
    if selected_job == -1:
        # 没有可调度的作业
        return -1

    # 3) 将 "机器选择规则" 映射到一个索引，比如 'FIFO' -> 0, 'SWT' -> 1
    # 你也可以直接用字符串做判断，而无需映射成 int。
    if mch_rule == 'FIFO':
        mch_rule_index = 0
    elif mch_rule == 'SWT':
        mch_rule_index = 1
    else:
        print(f'[Error] Unknown machine rule: {mch_rule}')
        sys.exit(1)

    # 4) 根据机器选择规则，选机器
    #    注意：select_machine 的参数 selected_op 一般是 (job * num_machines + ?)
    #    但是你在 select_machine 内部又把它拆成 j = selected_op // M。
    #    所以这里先把选定的 job_id 拼出一个合适的 op 索引再传过去。
    # selected_op = selected_job * env.number_of_machines  # 先加个 0 机器占位
    selected_mch = select_machine(env, e, mch_rule_index, selected_job)
    if selected_mch is None:
        return -1

    # 5) 组合成 action
    action = selected_job * env.number_of_machines + selected_mch
    return action



"""
    common utils
"""


def save_default_params(config):
    """
        save parameters in the config
    :param config: a package of parameters
    :return:
    """
    with open('./config_default.json', 'wt') as f:
        json.dump(vars(config), f, indent=4)
    print("successfully save default params")


def nonzero_averaging(x):
    """
        remove zero vectors and then compute the mean of x
        (The deleted nodes are represented by zero vectors)
    :param x: feature vectors with shape [sz_b, node_num, d]
    :return:  the desired mean value with shape [sz_b, d]
    """
    b = x.sum(dim=-2)
    y = torch.count_nonzero(x, dim=-1)
    z = (y != 0).sum(dim=-1, keepdim=True)
    p = 1 / z
    p[z == 0] = 0
    return torch.mul(p, b)


def strToSuffix(str):
    if str == '':
        return str
    else:
        return '+' + str


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    print('123')
