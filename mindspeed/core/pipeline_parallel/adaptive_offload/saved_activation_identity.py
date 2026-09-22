import weakref

import torch


def saved_tensor_metadata(tensor):
    try:
        if tensor.layout != torch.strided:
            return None
        return (tensor._version, tensor.device, tensor.dtype,
                tensor.untyped_storage().data_ptr(), tensor.storage_offset(),
                tuple(tensor.shape), tuple(tensor.stride()))
    except (RuntimeError, NotImplementedError):
        return None


class SavedActivationIdentity:
    def __init__(self):
        self.sources = {}
        self.records = {}

    def record(self, tag, tensor, witness=None):
        metadata = saved_tensor_metadata(tensor)
        if metadata is None:
            return
        source = self.sources.get(id(tensor))
        if source is None or source[0]() is not tensor or source[1] != metadata:
            source = (weakref.ref(tensor), metadata, tag)
            self.sources[id(tensor)] = source
        self.records[tag] = (source[2], metadata, weakref.ref(tensor if witness is None else witness))

    def aliases(self):
        canonical_tags = {}
        current_metadata = {}
        aliases = {}
        for tag, (identity, saved_metadata, reference) in self.records.items():
            tensor = reference()
            if tensor is None:
                continue
            tensor_id = id(tensor)
            if tensor_id not in current_metadata:
                current_metadata[tensor_id] = saved_tensor_metadata(tensor)
            if current_metadata[tensor_id] != saved_metadata:
                continue
            canonical = canonical_tags.setdefault(identity, tag)
            if canonical != tag:
                aliases[tag] = canonical
        return aliases

    def storage_footprint(self, expected_count):
        if len(self.records) != expected_count:
            return None
        storages = {}
        for _, _, reference in self.records.values():
            tensor = reference()
            if tensor is None:
                return None
            try:
                storage = tensor.untyped_storage()
                key = (tensor.device, storage._cdata, storage.data_ptr())
                nbytes = storage.nbytes()
                if tensor.device.type == "npu":
                    import torch_npu

                    nbytes = max(nbytes, torch_npu.get_storage_size(tensor) * tensor.element_size())
                storages[key] = max(storages.get(key, 0), nbytes)
            except (AttributeError, ImportError, RuntimeError, NotImplementedError):
                return None
        return storages

    def contiguous_copy_bytes(self, expected_count):
        if len(self.records) != expected_count:
            return None
        aliases = self.aliases()
        nbytes = 0
        for tag, (_, _, reference) in self.records.items():
            if tag in aliases:
                continue
            tensor = reference()
            if tensor is None:
                return None
            try:
                materialize = not tensor.is_contiguous()
                if tensor.device.type == "npu":
                    import torch_npu

                    materialize |= torch_npu.get_npu_format(tensor) != 2
                if materialize:
                    nbytes += tensor.numel() * tensor.element_size()
            except (AttributeError, ImportError, RuntimeError, NotImplementedError):
                return None
        return nbytes
