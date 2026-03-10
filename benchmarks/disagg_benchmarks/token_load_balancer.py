#!/usr/bin/env python3
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import ctypes

import requests
from transformers import AutoTokenizer

class AsyncTokenLoadBalancer:
    HISTORY_TOKEN_LIMIT = 2**63 - 10000
    def __init__(self, model: str, prefill_clients: List[Dict[str, Any]]):
        # ⚠️HINT: 用单例避免重复加载
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        self._executor = ThreadPoolExecutor(max_workers=4)
        self.lock = asyncio.Lock()
        self.prefill_clients = prefill_clients
        assert len(prefill_clients) > 0, "Prefill clients should not be empty."

        self.keys = [f"{c['host']}:{c['port']}:id{c['id']}" for c in prefill_clients]
        self.clients = {k: c for k, c in zip(self.keys, prefill_clients)}
        # 记录每个后端当前 token 总量
        self.loader: Dict[str, int] = {k: 0 for k in self.keys}
        self.loader = {key: {'token_num': int(0), 'requests': {}, 'history_tokens': int(0)} for key in self.keys}

    # ---------- 内部计数 ----------
    def _count_chat_tokens(self, messages: List[Dict[str, str]]) -> int:
        total = 3  # openai 官方偏移
        for m in messages:
            total += len(self.tokenizer.encode(m["role"], add_special_tokens=False))
            total += len(self.tokenizer.encode(m["content"], add_special_tokens=False))
        return total

    def _count_completion_tokens(self, prompt) -> int:
        if isinstance(prompt, list):
            prompt = "\n".join(prompt)
        return len(self.tokenizer.encode(prompt, add_special_tokens=True))

    def _parse_prompt_length(self, request: Dict[str, Any]) -> int:
        # loop = asyncio.get_running_loop()
        if "messages" in request:
            # return loop.run_in_executor(
            #     self._executor, self._count_chat_tokens, request["messages"]
            # )
            return self._count_chat_tokens(request["messages"])
        if "prompt" in request:
            # return loop.run_in_executor(
            #     self._executor, self._count_completion_tokens, request["prompt"]
            # )
            return self._count_completion_tokens(request["prompt"])
        assert False, "Unknown request format, neither 'messages' nor 'prompt' found."
        return 0

    async def analyze(self, request_id: str, request: dict):
        token_num = self._parse_prompt_length(request)
        if len(self.loader) == 0:
          key = self.keys[0]
          async with self.lock:
            self.loader[key]['token_num'] += token_num
            self.loader[key]['history_tokens'] += token_num
            self.loader[key]['requests'][request_id] = token_num
          return self.clients[key]

        async with self.lock:
            min_key = min(self.loader.keys(), key=lambda k: self.loader[k]['token_num'])
            if self.loader[min_key]['token_num'] == 0:
                min_key = min(self.loader.keys(), key=lambda k: self.loader[k]['history_tokens'])
            if self.loader[min_key]['history_tokens'] + token_num > AsyncTokenLoadBalancer.HISTORY_TOKEN_LIMIT:
                # all backend exceeded history limit, reset all
                for k in self.loader.keys():
                    self.loader[k]['history_tokens'] = ctypes.c_uint64(0)
                min_key = min(self.loader.keys(), key=lambda k: self.loader[k]['token_num'])
            else:
                self.loader[min_key]['history_tokens'] += token_num
            self.loader[min_key]['token_num'] += token_num
            self.loader[min_key]['requests'][request_id] = token_num
        return self.clients[min_key]

    async def release(self, request_id: str):
        async with self.lock:
            for key in self.loader:
                if request_id in self.loader[key]['requests']:
                    print("releasing request_id:", request_id)
                    self.loader[key]['token_num'] -= self.loader[key]['requests'][request_id]
                    del self.loader[key]['requests'][request_id]