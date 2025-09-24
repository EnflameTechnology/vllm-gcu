#!/usr/bin/env python
# coding=utf-8
import sys
from vllm.v1.engine.core_client import DPLBAsyncMPClient, logger
from unittest.mock import patch


class PatchedDPLBAsyncMPClient(DPLBAsyncMPClient):

    def get_core_engine_for_request(self, request):
        # Engines are in rank order.
        if (eng_index := request.data_parallel_rank) is None:
            if not self.lb_engines:
                return self.core_engine
            # TODO use P2C alg for larger DP sizes
            num_engines = len(self.lb_engines)
            min_counts = [sys.maxsize, sys.maxsize]
            eng_index = 0
            for i in range(num_engines):
                # Start from client_index to help with balancing when engines
                # are empty.
                idx = (self.client_index + i) % num_engines
                counts = self.lb_engines[idx]
                if sum(counts) < sum(min_counts):
                    min_counts = counts
                    eng_index = idx
            # Adjust local counts for better balancing between stats updates
            # from the coordinator (which happen every 100ms).
            if min_counts[0]:
                min_counts[0] += 1
            else:
                min_counts[1] += 1

        chosen_engine = self.core_engines[eng_index]
        # Record which engine is chosen for this request, to handle aborts.
        self.reqs_in_flight[request.request_id] = chosen_engine
        logger.info(
            f"request id: {request.request_id}, chosen_engine: {chosen_engine}, lb_engines: {self.lb_engines}"
        )
        return chosen_engine


patch("vllm.v1.engine.core_client.DPLBAsyncMPClient",
      PatchedDPLBAsyncMPClient).start()
