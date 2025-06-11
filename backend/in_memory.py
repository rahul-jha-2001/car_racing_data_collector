# in_memory_store.py
import asyncio
from collections import defaultdict, deque

class InMemoryStore:
    def __init__(self):
        self.store = defaultdict(deque)
        self.expiry_tasks = {}

    async def rpush(self, key, value):
        self.store[key].append(value)

    async def lrange(self, key, start, end):
        return list(self.store[key])[start:end + 1 if end != -1 else None]

    async def expire(self, key, ttl):
        if key in self.expiry_tasks:
            self.expiry_tasks[key].cancel()

        self.expiry_tasks[key] = asyncio.create_task(self._expire_after(key, ttl))

    async def _expire_after(self, key, ttl):
        await asyncio.sleep(ttl)
        self.store.pop(key, None)
        self.expiry_tasks.pop(key, None)

    async def delete(self, key):
        self.store.pop(key, None)

    async def exists(self, key):
        return key in self.store
