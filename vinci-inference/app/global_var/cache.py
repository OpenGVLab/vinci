import threading
from typing import OrderedDict


class FIFOSafeCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key not in self.cache:
                return None
            # 移动到末尾以保持顺序
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                # 如果键已存在，更新值并移动到末尾
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                # 移除第一个（最旧的）条目
                self.cache.popitem(last=False)