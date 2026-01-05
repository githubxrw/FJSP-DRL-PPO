import numpy as np
import copy
import sys
import torch
from common_utils import min_element_index, max_element_index
from params import configs

config = configs

class PriorityDispatchingRules:
    """
    Class containing various Priority Dispatching Rules (PDRs) for FJSP.
    """

    def __init__(self, env, flag=False):
        """
        Initialize with the environment.

        :param env: FJSPEnvForSameOpNums instance to access environment features.
        """
        self.env = env  # Access to environment features like op_mean_pt, etc.
        self.flag = flag

    def PDR_rules(self, rule_op, rule_mch):
        selected_ops = []
        selected_mchs = []

        for e in range(self.env.number_of_envs):

            # 1) sampling 并行环境：已完成的 env 必须给占位动作
            if self.flag and self.env.done[e]:
                selected_ops.append(0)
                selected_mchs.append(0)
                continue

            op_rule = int(rule_op[e])
            chosen_job = self.select_operation(e, op_rule)
            if chosen_job == -1:
                # 2) 选不到可调度作业：也必须给占位动作，避免后续数组错位/非法
                selected_ops.append(0)
                selected_mchs.append(0)
                continue

            mch_rule = int(rule_mch[e])
            chosen_mch = self.select_machine(e, mch_rule, chosen_job)
            if chosen_mch is None:
                selected_ops.append(0)
                selected_mchs.append(0)
                continue

            selected_ops.append(chosen_job)
            selected_mchs.append(chosen_mch)

        return np.array(selected_ops, dtype=np.int64), np.array(selected_mchs, dtype=np.int64)

    def select_operation(self, e, rule_op):
        """
        Select operation based on the specified PDR rule, considering dynamic_pair_mask.

        :param e: Environment index
        :param rule_op: Operation PDR rule index
        :return: selected_op or None if no valid operation
        """

        # Modify this part to use heuristic_select_action_for_job instead of previous logic
        # Choose job based on the selected rule
        selected_op = self.select_job_based_on_rule(rule_op, e)

        return selected_op

    def select_job_based_on_rule(self, rule_op, e):
        """
        Select job based on the heuristic method passed in the rule_op.

        :param rule_op: The heuristic rule (FIFO, MOR, etc.)
        :param e: Environment index
        :param candidate_ops: Candidate operations (array)
        :param dynamic_pair_mask: Pair mask indicating valid operations
        :return: The selected job (index)
        """
        # Heuristic job selection logic (based on modified job selection rules)
        if rule_op == 0:  # FIFO (First In First Out)
            return self.heuristic_select_action_for_job('FIFO', self.env, e)

        elif rule_op == 1:  # SPT
            return self.heuristic_select_action_for_job('SPT', self.env, e)

        elif rule_op == 2:  # MWKR (Most Work Remaining)
            return self.heuristic_select_action_for_job('MWKR', self.env, e)

        elif rule_op == 3:  # MOPNR (Most Operations Remaining)
            return self.heuristic_select_action_for_job('MOR', self.env, e)
        elif rule_op == 4:
            return self.heuristic_select_action_for_job('RANDOM', self.env, e)

        else:
            raise ValueError(f"Unknown operation PDR rule index: {rule_op}")

    def heuristic_select_action_for_job(self, method, env, e):
        """
        :param method: the name of heuristic method
        :param env: the environment
        :return: the job selected by the heuristic method

        here are heuristic methods selected for comparison:

        FIFO: First in first out
        MOR(or MOPNR): Most operations remaining
        SPT: Shortest processing time
        MWKR: Most work remaining
        """
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

            # 仅在可调度作业 available_jobs 上计算：每个 job 在“可行且可用机器集合”上的最小 PT

            if len(available_jobs) == 0:
                return -1

            # env.candidate_pt[e] 期望形状: [J, M]

            # env.dynamic_pair_mask[e] 期望形状: [J, M]，True 表示该 pair 不可选（时间或工艺原因）

            temp_pt = copy.deepcopy(env.candidate_pt[e])  # [J, M]

            temp_pt[env.dynamic_pair_mask[e]] = float("inf")

            # 对每个 job 取 min PT（inf 表示该 job 当前无任何可用 pair）

            job_min_pt = np.min(temp_pt, axis=1)  # [J]

            # 只考虑 available_jobs

            cand_min_pt = job_min_pt[available_jobs]

            best_val = np.min(cand_min_pt)

            if np.isinf(best_val):
                # available_jobs 中没有任何 job 有可行可用机器

                return -1

            best_jobs = available_jobs[np.where(cand_min_pt == best_val)[0]]

            chosen_job = int(np.random.choice(best_jobs))

        elif method == 'MWKR':
            job_remain_work_list = env.op_match_job_remain_work[e][available_ops]
            # print(env.op_match_job_remain_work[e][available_ops])
            chosen_job = available_jobs[np.random.choice(max_element_index(job_remain_work_list)[0])]

        elif method == 'RANDOM':
            if len(available_jobs) == 0:
                return -1  # 无可用作业
            chosen_job = np.random.choice(available_jobs)

        else:
            print(f'Error From rule select: undefined method {method}')
            sys.exit()

        if chosen_job == -1:
            print(f'Error From choosing job: choose job {chosen_job}')
            sys.exit()

        return chosen_job

    def select_jobs_by_rule(self, rule_op_np):
        # === 关键：done_mask 来自 env.done()（ndarray），而不是 env.done 属性本身 ===
        done_attr = getattr(self.env, "done", None)
        done_mask = done_attr() if callable(done_attr) else done_attr  # ndarray / bool / None

        # 安全检查：done_mask 若是 ndarray，应与 number_of_envs 对齐
        if isinstance(done_mask, np.ndarray) and done_mask.shape[0] != self.env.number_of_envs:
            print("[Debug] done_mask.shape =", done_mask.shape,
                  "number_of_envs =", self.env.number_of_envs)
            raise ValueError("done_mask length mismatch with number_of_envs")

        jobs = np.zeros(self.env.number_of_envs, dtype=np.int64)
        for e in range(self.env.number_of_envs):
            # === 必做：done env 占位动作 ===
            if isinstance(done_mask, np.ndarray):
                if done_mask[e]:
                    jobs[e] = 0
                    continue
            elif isinstance(done_mask, (bool, np.bool_)):
                if bool(done_mask):
                    jobs[e] = 0
                    continue

            r = int(rule_op_np[e])
            j = self.select_job_based_on_rule(r, e)

            if j == -1:
                feasible = np.where((self.env.mask[e] == 0) &
                                    (np.any(~self.env.dynamic_pair_mask[e], axis=1)))[0]
                j = int(feasible[0]) if len(feasible) > 0 else 0
            jobs[e] = j
        return jobs

    def select_mchs_by_rule(self, rule_mch_np, chosen_job_np):
        done_attr = getattr(self.env, "done", None)
        done_mask = done_attr() if callable(done_attr) else done_attr

        if isinstance(done_mask, np.ndarray) and done_mask.shape[0] != self.env.number_of_envs:
            print("[Debug] done_mask.shape =", done_mask.shape,
                  "number_of_envs =", self.env.number_of_envs)
            raise ValueError("done_mask length mismatch with number_of_envs")

        mchs = np.zeros(self.env.number_of_envs, dtype=np.int64)
        for e in range(self.env.number_of_envs):
            # === 必做：done env 占位动作 ===
            if isinstance(done_mask, np.ndarray):
                if done_mask[e]:
                    mchs[e] = 0
                    continue
            elif isinstance(done_mask, (bool, np.bool_)):
                if bool(done_mask):
                    mchs[e] = 0
                    continue

            r = int(rule_mch_np[e])
            j = int(chosen_job_np[e])

            # 当前可用机器集合：优先 dynamic_pair_mask（工艺+时间）
            avail = np.where(~self.env.dynamic_pair_mask[e, j])[0]
            if len(avail) == 0:
                avail = np.where(~self.env.candidate_process_relation[e, j])[0]
            if len(avail) == 0:
                mchs[e] = 0
                continue

            if r == 0:
                ft = self.env.mch_free_time[e, avail]
                mchs[e] = int(avail[np.argmin(ft)])
            elif r == 1:
                wt = self.env.mch_waiting_time[e, avail]
                mchs[e] = int(avail[np.argmin(wt)])
            else:
                mchs[e] = int(np.random.choice(avail))
        return mchs

    def select_machine(self, e, rule_mch, selected_op):
        """
        Select machine based on the specified PDR rule, considering dynamic_pair_mask for the selected operation.

        :param e: Environment index
        :param rule_mch: Machine PDR rule index
        :param selected_op: Previously selected operation
        :return: selected_mch or None if no valid machine
        """
        env = self.env
        M = self.env.number_of_machines

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
            # selected_mch = compatible_mchs[np.argmax(self.env.mch_free_time[e, compatible_mchs])]
        elif rule_mch == 1:  # SWT
            # 找到可处理选定操作的机器
            mch_state = ~env.candidate_process_relation[e, selected_op]
            available_mchs = np.where(mch_state == True)[0]

            # 获取可用机器的等待时间
            mch_wait_time = env.mch_waiting_time[e][available_mchs]
            # 找到最大等待时间
            max_wait_time = np.min(mch_wait_time)
            # 找到所有等待时间等于最大值的机器
            chosen_mch_list = available_mchs[mch_wait_time == max_wait_time]
            # 随机选择一台
            selected_mch = np.random.choice(chosen_mch_list)
            return selected_mch
        elif rule_mch == 2:  # random
            mch_state = ~env.candidate_process_relation[e, selected_op]
            available_mchs = np.where(mch_state == True)[0]
            # print(available_mchs)
            if len(available_mchs) == 0:
                return None  # 无可用机器
            # available_mchs = np.where(mch_state == True)[0]
            # print(np.where(mch_state == True))
            selected_mch = np.random.choice(available_mchs)
        else:
            raise ValueError(f"Unknown machine PDR rule index: {rule_mch}")

        return selected_mch
