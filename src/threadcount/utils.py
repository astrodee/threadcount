from types import MethodType


def get_custom_attr(top, cls):
    """pass attributes from one class to another"""
    attr_names = dir(cls)
    for name in attr_names:
        attr = getattr(cls, name)
        if not (name.startswith("_") or type(attr) is MethodType):
            setattr(top, name, attr)
