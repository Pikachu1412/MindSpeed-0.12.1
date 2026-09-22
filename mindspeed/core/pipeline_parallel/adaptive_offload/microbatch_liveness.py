import weakref
from collections import Counter

import torch


class _OutputLifetimeHook:
    def __init__(self, tracker, token, index):
        self.tracker = tracker
        self.token = token
        self.index = index

    def __call__(self, gradient):
        self.tracker.backward_started(self.token, self.index)
        return gradient


class MicrobatchLiveness:
    def __init__(self, record_count):
        self.record_count = record_count
        self.live = set()
        self.records = {}
        self.counts = Counter()
        self.peak_live = 0
        self.peak_backward = 0

    def _sample(self):
        self.peak_live = max(self.peak_live, len(self.live))
        self.peak_backward = max(self.peak_backward, sum(bool(record['tasks']) for record in self.records.values()))
        self.record_count(len(self.live))

    def start_iteration(self):
        if any(record['tasks'] for record in self.records.values()):
            raise RuntimeError('Cannot start a new activation iteration with unfinished autograd tasks')
        self.counts = Counter(carried_graphs=len(self.live))
        self.peak_live = len(self.live)
        self.peak_backward = 0
        self._sample()

    def begin(self, token):
        if token in self.records:
            raise RuntimeError('Duplicate activation microbatch token')
        self.live.add(token)
        self.records[token] = {'outputs': {}, 'tasks': {}, 'attached': False, 'started': False}
        self.counts['forward_started'] += 1
        self._sample()

    def track(self, token, output):
        record = self.records[token]
        if record['attached']:
            raise RuntimeError('Microbatch outputs were already attached')
        tensors = {}
        pending = [output]
        while pending:
            value = pending.pop()
            if isinstance(value, torch.Tensor) and value.requires_grad:
                tensors[id(value)] = value
            elif isinstance(value, (tuple, list)):
                pending.extend(reversed(value))
            elif isinstance(value, dict):
                pending.extend(reversed(tuple(value.values())))
        for index, tensor in enumerate(tensors.values()):
            record['outputs'][index] = {'pending': set(), 'complete': False, 'collected': False, 'handle': None}
            hook = _OutputLifetimeHook(self, token, index)
            weakref.finalize(hook, self.output_collected, token, index)
            record['outputs'][index]['handle'] = tensor.register_hook(hook)
        record['attached'] = True
        self._maybe_retire(token)
        return output

    def backward_started(self, token, index):
        record = self.records.get(token)
        if record is None:
            return
        graph_task = torch._C._current_graph_task_id()
        keep_graph = torch._C._autograd._get_current_graph_task_keep_graph()
        if graph_task < 0:
            raise RuntimeError('Activation backward hook ran outside an autograd task')
        if not record['started']:
            record['started'] = True
            self.counts['backward_started'] += 1
        if graph_task not in record['tasks']:
            record['tasks'][graph_task] = {'outputs': set(), 'keep_graph': keep_graph}
            torch.autograd.Variable._execution_engine.queue_callback(
                lambda: self.backward_completed(token, graph_task)
            )
        record['tasks'][graph_task]['outputs'].add(index)
        record['outputs'][index]['pending'].add(graph_task)
        self._sample()

    def backward_completed(self, token, graph_task):
        record = self.records[token]
        task = record['tasks'].pop(graph_task)
        self.counts['backward_tasks_completed'] += 1
        for index in task['outputs']:
            output = record['outputs'][index]
            output['pending'].remove(graph_task)
            output['complete'] |= not task['keep_graph'] or output['collected']
            if output['complete'] and not output['pending']:
                handle = output['handle']
                output['handle'] = None
                if handle is not None:
                    handle.remove()
        self._maybe_retire(token)

    def output_collected(self, token, index):
        record = self.records.get(token)
        if record is None:
            return
        output = record['outputs'][index]
        output['collected'] = True
        self.counts['output_hooks_collected'] += 1
        self._maybe_retire(token)

    def _maybe_retire(self, token):
        record = self.records.get(token)
        if record is None or not record['attached'] or record['tasks']:
            return
        if not all(output['complete'] or output['collected'] for output in record['outputs'].values()):
            return
        del self.records[token]
        self.live.remove(token)
        self.counts['microbatches_retired'] += 1
        if record['started']:
            self.counts['backward_microbatches_completed'] += 1
        self._sample()

    def snapshot(self):
        backward = sum(bool(record['tasks']) for record in self.records.values())
        waiting = sum(not record['started'] for record in self.records.values())
        return {**dict(self.counts), 'live': len(self.live), 'backward': backward, 'awaiting_backward': waiting,
                'retained_after_backward': len(self.live) - backward - waiting,
                'peak_live': self.peak_live, 'peak_backward': self.peak_backward}
