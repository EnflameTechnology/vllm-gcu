# SPDX-License-Identifier: Apache-2.0
from vllm.core.scheduler import (
    Scheduler,
    SchedulerOutputs,
    SchedulerPrefillOutputs,
    SchedulerRunningOutputs,
    SchedulerSwappedInOutputs,
    SchedulingBudget,
)
from vllm.sequence import SequenceStatus


class PriorityScheduler(Scheduler):
    """priority scheduler for V0"""

    def _schedule_default(self) -> SchedulerOutputs:
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        for seq_group in self.running:
            budget.add_num_seqs(
                seq_group.request_id, seq_group.get_max_num_running_seqs()
            )
        curr_loras = (
            set(
                seq_group.lora_int_id
                for seq_group in self.running
                if seq_group.lora_int_id > 0
            )
            if self.lora_enabled
            else None
        )

        prefills = SchedulerPrefillOutputs.create_empty()
        running_scheduled = SchedulerRunningOutputs.create_empty()
        swapped_in = SchedulerSwappedInOutputs.create_empty()

        if not self.swapped:
            prefills = self._schedule_prefills(
                budget, curr_loras, enable_chunking=False
            )

        if self.scheduler_config.policy == "priority":
            self._schedule_priority_preemption(budget)

        count = 0
        while count < len(prefills.seq_groups):
            seq_group = prefills.seq_groups[count]
            if len(self.running) == 0:
                break
            if (
                self._get_priority(self.running[-1])[0]
                < self._get_priority(seq_group.seq_group)[0]
            ):
                # all decode reqs are prior to new prefill, add prefill to waiting
                num_running_tokens_uncached, _ = (
                    self._get_num_new_uncached_and_cached_tokens(
                        seq_group.seq_group, SequenceStatus.RUNNING, False, budget
                    )
                )
                budget.subtract_num_batched_tokens(
                    seq_group.seq_group.request_id, num_running_tokens_uncached
                )
                num_running_seqs = seq_group.seq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(
                    seq_group.seq_group.request_id, num_running_seqs
                )
                seqs = seq_group.seq_group.get_seqs(status=SequenceStatus.RUNNING)
                for seq in seqs:
                    seq.status = SequenceStatus.WAITING
                    self.free_seq(seq)
                self.waiting.extend([seq_group.seq_group])
                prefills.seq_groups.remove(seq_group)
            else:
                count += 1

        if len(prefills.seq_groups) == 0:
            running_scheduled = self._schedule_running(
                budget, curr_loras, enable_chunking=False
            )

            if (
                len(running_scheduled.preempted) + len(running_scheduled.swapped_out)
                == 0
            ):
                swapped_in = self._schedule_swapped(budget, curr_loras)

        assert budget.num_batched_tokens <= self.scheduler_config.max_num_batched_tokens
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.waiting.extendleft(running_scheduled.preempted)
        # Update new running requests.
        if len(prefills.seq_groups) > 0:
            self.running.extend([s.seq_group for s in prefills.seq_groups])

        self.running.extend(running_scheduled.decode_seq_groups_list)

        if len(swapped_in.decode_seq_groups) > 0:
            self.running.extend([s.seq_group for s in swapped_in.decode_seq_groups])

        # Update swapped requests.
        self.swapped.extend(running_scheduled.swapped_out)
        preempted = len(running_scheduled.preempted) + len(
            running_scheduled.swapped_out
        )

        # There should be no prefill from running queue because this policy
        # doesn't allow chunked prefills.
        assert len(running_scheduled.prefill_seq_groups) == 0
        assert len(swapped_in.prefill_seq_groups) == 0

        # Merge lists
        num_prefill_groups = len(prefills.seq_groups)
        if num_prefill_groups > 0:
            scheduled_seq_groups = prefills.seq_groups
            scheduled_seq_groups.extend(running_scheduled.decode_seq_groups)
        else:
            scheduled_seq_groups = running_scheduled.decode_seq_groups
        scheduled_seq_groups.extend(swapped_in.decode_seq_groups)

        blocks_to_copy = running_scheduled.blocks_to_copy
        blocks_to_copy.extend(swapped_in.blocks_to_copy)

        ignored_seq_groups = prefills.ignored_seq_groups
        ignored_seq_groups.extend(swapped_in.infeasible_seq_groups)

        return SchedulerOutputs(
            scheduled_seq_groups=scheduled_seq_groups,
            num_prefill_groups=num_prefill_groups,
            num_batched_tokens=budget.num_batched_tokens + budget.num_cached_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=preempted,
        )
