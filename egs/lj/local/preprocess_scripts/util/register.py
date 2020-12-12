import collections


def register(input_type, register_dict):
    def decorator(func):
        nonlocal input_type, register_dict
        if isinstance(input_type, str):
            input_type = [input_type]
        elif not isinstance(input_type, collections.Iterable):
            raise ValueError('input_type only support str or iterable')

        for in_t in input_type:
            register_dict[in_t] = func

        def get_input(*args, **kwargs):
            return func(*args, **kwargs)

        return get_input

    return decorator

