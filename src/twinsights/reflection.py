from importlib import import_module


def get_class_for_name(class_name: str):
    try:
        module_path, class_name = class_name.rsplit('.', 1)
        module = import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(class_name)
