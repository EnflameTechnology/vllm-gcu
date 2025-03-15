# !/usr/bin/env python
# coding=utf-8
import asyncio
import itertools
import os
from argparse import Namespace
from typing import List

import aiohttp
from aiohttp import web
from vllm.logger import init_logger


logger = init_logger("vllm_gcu.router")


async def send_idle_request(url, model):
    async with aiohttp.ClientSession() as session:
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
        payload = {
            "model": model,
            "prompt": "hi",
            "temperature": 0.0,
            "max_tokens": 1,
            "stream": True,
            "stream_options": {"include_usage": True},
            "priority": 100,
            "ignore_eos": True,
        }

        try:
            async with session.post(
                url=url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                    return url
        except Exception:
            return url


class RoundRobinProxy:
    def __init__(
        self, server_urls: List[str], model: str, entrypoint: str = "/v1/completions"
    ):
        self.server_cycle = itertools.cycle(server_urls)
        self.active_idle_loop = asyncio.Event()
        self.requests = 0
        self.lock = asyncio.Lock()

        asyncio.create_task(self.idle_loop(server_urls, model, entrypoint))

    async def increment(self):
        async with self.lock:
            self.requests += 1
            if self.requests == 1:
                self.active_idle_loop.set()

    async def decrement(self):
        async with self.lock:
            if self.requests > 0:
                self.requests -= 1

            if self.requests == 0:
                self.active_idle_loop.clear()

    async def idle_loop(
        self, server_urls: List[str], model: str, entrypoint: str = "/v1/completions"
    ):
        while True:
            await self.active_idle_loop.wait()

            idle_tasks = [
                asyncio.create_task(send_idle_request(f"{url}{entrypoint}", model))
                for url in server_urls
            ]

            while self.requests > 0:
                done, _ = await asyncio.wait(
                    idle_tasks, return_when=asyncio.FIRST_COMPLETED
                )

                for d in done:
                    server_url = d.result()
                    idle_tasks.remove(d)
                    idle_tasks.append(
                        asyncio.create_task(send_idle_request(server_url, model))
                    )

            for t in idle_tasks:
                t.cancel()

    async def handle_request(self, request):
        target_url = f"{next(self.server_cycle)}{request.path_qs}"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.request(
                    method=request.method,
                    url=target_url,
                    headers=request.headers,
                    data=request.content,
                ) as response:

                    await self.increment()

                    resp = web.StreamResponse(
                        status=response.status, headers=response.headers
                    )

                    await resp.prepare(request)

                    async for chunk in response.content.iter_any():
                        await resp.write(chunk)

                    await resp.write_eof()

                    await self.decrement()
                    return resp

            except Exception as e:
                return web.Response(text=f"Error: {str(e)}", status=500)


async def router(args: Namespace):
    proxy = RoundRobinProxy(args.server_urls, args.model)
    app = web.Application()
    app.router.add_route("*", "/{path:.*}", proxy.handle_request)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.port)
    await site.start()

    logger.info(f"Starting vLLM DP Router on http://{args.host}:{args.port}")

    await asyncio.Event().wait()


if __name__ == "__main__":
    from vllm.utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--server-urls", type=str, default=None, nargs="+")
    parser.add_argument("--entrypoints", type=str, default="/v1/completions")
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    asyncio.run(router(args))

