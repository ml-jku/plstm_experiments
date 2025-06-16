from .compose_transform import ComposeTransform
from .single_factory import SingleFactory


class TransformFactory(SingleFactory):
    def create(self, obj_or_kwargs, collate_fn=None, **kwargs):
        if isinstance(obj_or_kwargs, list):
            assert collate_fn is None
            return super().create(obj_or_kwargs, collate_fn=ComposeTransform, **kwargs)
        return super().create(obj_or_kwargs, **kwargs)
