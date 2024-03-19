


from collections.abc import Iterable
from functools import wraps

def build_condition_func(func=None, keys=None, types=None, values=None):
    """
    Builds a list of condition functions based on the provided parameters.

    Parameters:
    - func (callable, optional): A function that takes an element of the iterable and returns True or False.
    - keys (iterable, optional): A list of keys. If the data is a dictionary and contains all the keys, the condition function returns True.
    - types (type or iterable of types, optional): A type or a list of types. If the data is an instance of any of the types, the condition function returns True.
    - values (iterable, optional): A list of values. If the data is equal to any of the values, the condition function returns True.

    Returns:
    A list of condition functions that can be used to filter elements based on the provided parameters.

    Example:
    condition_funcs = build_condition_funcs(types=int)
    # The condition_funcs will contain a function that returns True if the input data is of type int
    """
    if func is None and keys is None and types is None and values is None:
        def default_condition(data):
            return not isinstance(data, (dict, list, tuple))
        func = default_condition

    condition_funcs = []
    if func is not None:
        condition_funcs.append(func)
    if keys is not None:
        if not isinstance(keys, Iterable) or isinstance(keys, str):
            keys = [keys]
        key_set = set(keys)
        condition_funcs.append(lambda data: isinstance(data, dict) and key_set.issubset(set(data.keys())))
    if types is not None:
        if isinstance(types, type):
            types = [types]
        condition_funcs.append(lambda data: isinstance(data, tuple(types)))
    if values is not None:
        condition_funcs.append(lambda data: data in values)

    def condition_func(data):
        try:
            return all(condition(data) for condition in condition_funcs)
        except:
            return False

    return condition_func




def flatten(data, func=None, keys=None, types=None, values=None):
    """
    Flattens a nested data structure (dict/list) into a list containing elements that match the specified conditions.

    Parameters:
    - data (dict/list): The input nested data structure.
    - func (callable, optional): A function that takes an element of the iterable and returns True or False.
    - keys (iterable, optional): A list of keys. If the data is a dictionary and contains all the keys, the element is included in the output list.
    - types (type or iterable of types, optional): A type or a list of types. If the data is an instance of any of the types, the element is included in the output list.
    - values (iterable, optional): A list of values. If the data is equal to any of the values, the element is included in the output list.

    Returns:
    A list containing elements from the input data that match the specified conditions.

    Example:
    data = {"foo": [1, 2], "bar": {"baz": 3}}
    result = flatten(data, types=int)
    # result: [1, 2, 3]
    """
    condition_func = build_condition_func(func, keys, types, values)

    def _flatten(data):
        if condition_func(data):
            return [data]
        if isinstance(data, dict):
            result = []
            for v in data.values():
                result.extend(_flatten(v))
            return result
        elif isinstance(data, list) or isinstance(data, tuple):
            result = []
            for i in data:
                result.extend(_flatten(i))
            return result
        else:
            return []

    return _flatten(data)




def elementwise_apply(func=None, keys=None, types=None, values=None):
    """
    A decorator that applies a function to certain elements of an iterable.

    Parameters:
    - func (callable): a function that takes an element of the iterable and returns True or False.
    - keys (iterable): a list of keys. If the data is a dictionary and contains all the keys, the decorator is applied.
    - types (type or iterable of types): a type or a list of types. If the data is an instance of any of the types, the decorator is applied.
    - values (iterable): a list of values. If the data is equal to any of the values, the decorator is applied.

    Returns:
    A decorated function that applies the original function to certain elements of the input data.

    Usage:
    @elementwise_apply
    def my_func(data):
        # do something to data
        return processed_data

    my_data = {"foo": [1, 2], "bar": {"baz": 3}}
    processed_data = my_func(my_data)

    The `my_func` function will be applied to `my_data["foo"][0]`, `my_data["foo"][1]`, and `my_data["bar"]["baz"]`. The resulting data will have the same structure as `my_data`.

    Note that this function only works on nested structures that are iterable (e.g. lists, tuples, and dictionaries). Non-iterable values (e.g. integers, strings, and other scalar types) are not modified by this decorator.
    """
    condition_func = build_condition_func(func=func, keys=keys, types=types, values=values)
    
    def apply_func_decorator(wrap_func):
        @wraps(wrap_func)
        def wrapper(data, *args, **kwargs):
            if condition_func(data):
                return wrap_func(data, *args, **kwargs)
        
            if isinstance(data, dict):
                return {k: wrapper(v, *args, **kwargs) for k, v in data.items()}
            elif isinstance(data, list):
                return [wrapper(i, *args, **kwargs) for i in data]
            elif isinstance(data, tuple):
                return tuple(wrapper(i, *args, **kwargs) for i in data)
            else:
                return data
        
        return wrapper
    return apply_func_decorator



import unittest

class TestFlatten(unittest.TestCase):
    def test_flatten_list(self):
        data = [1, 2, 3, 4, 5]
        result = flatten(data, func=lambda x: x % 2 == 0)
        self.assertEqual(result, [2, 4])

    def test_flatten_dict(self):
        data = {"a": {"a": 1, "b": 2, "c": 3}, "b": {"a": 3, "b2": 4, "c": 7}}
        result = flatten(data, keys=["a", "c"])
        self.assertEqual(result, [{"a": 1, "b": 2, "c": 3}, {"a": 3, "b2": 4, "c": 7}])

    def test_flatten_nested_list(self):
        data = [1, [2, 3], 4, [5, [6, 7]]]
        result = flatten(data, func=lambda x: x % 2 == 0)
        self.assertEqual(result, [2, 4, 6])

    def test_flatten_nested_dict(self):
        data = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
        result = flatten(data, keys=["e"])
        self.assertEqual(result, [{"e": 3}])
    
    def test_flatten_nested_mixed(self):
        data = {"a": [1, 2], "b": {"c": [3, 4], "d": {"e": 5}}}
        result = flatten(data, func=lambda x: x % 2 == 0)
        self.assertEqual(result, [2, 4])

    def test_flatten_type_condition(self):
        data = [1, "a", 2, "b", {"c": 3}]
        result = flatten(data, types=str)
        self.assertEqual(result, ["a", "b"])

    def test_flatten_value_condition(self):
        data = [1, 2, 3, 4, 5]
        result = flatten(data, values=[1, 4])
        self.assertEqual(result, [1, 4])
        
    def test_flatten_empty_data(self):
        data = {}
        result = flatten(data, keys=["a", "c"])
        self.assertEqual(result, [])

    def test_flatten_varying_levels(self):
        data = {"a": {"a": 1, "b": 2, "c": {"a": 3, "c": 7}}, "b": [{"a": 3, "c": 7}]}
        result = flatten(data, keys=["a", "c"])
        self.assertEqual(result, [{"a": 1, "b": 2, "c": {"a": 3, "c": 7}}, {"a": 3, "c": 7}])


    def test_flatten_non_dict_elements(self):
        data = {"a": [{"a": 1, "c": 3}], "b": [1, 2, 3]}
        result = flatten(data, keys=["a", "c"])
        self.assertEqual(result, [{"a": 1, "c": 3}])

    def test_flatten_no_matches(self):
        data = {"a": {"a": 1, "b": 2}, "b": {"a": 3, "b2": 4}}
        result = flatten(data, keys=["a", "c"])
        self.assertEqual(result, [])
        
    def test_flatten_condition_funcs(self):
        data = {"a": {"a": 1, "b": 2, "c": 3}, "b": {"a": 3, "b2": 4, "c": 7}}

        def condition_func(data):
            return "b2" in data

        result = flatten(data, func=condition_func)
        self.assertEqual(result, [{"a": 3, "b2": 4, "c": 7}])



class Testelementwise_apply(unittest.TestCase):
    def test_elementwise_apply_with_func(self):
        @elementwise_apply(func=lambda x: isinstance(x, int))
        def add_one(x):
            return x + 1

        data = [1, '2', 3, '4', [5, '6', 7]]
        result = add_one(data)
        self.assertEqual(result, [2, '2', 4, '4', [6, '6', 8]])

    def test_elementwise_apply_with_keys(self):
        @elementwise_apply(keys=['a', 'b'])
        def process(d):
            d['a'] += 1
            d['b'] += 1
            return d

        data = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4, 'c': 5}]
        result = process(data)
        self.assertEqual(result, [{'a': 2, 'b': 3}, {'a': 4, 'b': 5, 'c': 5}])

    def test_elementwise_apply_with_types(self):
        @elementwise_apply(types=int)
        def add_one(x):
            return x + 1

        data = [1, '2', 3, '4', [5, '6', 7]]
        result = add_one(data)
        self.assertEqual(result, [2, '2', 4, '4', [6, '6', 8]])

    def test_elementwise_apply_with_values(self):
        @elementwise_apply(values={1, 2, 3})
        def add_one(x):
            return x + 1

        data = [1, 2, 3, 4, [5, 1, 2]]
        result = add_one(data)
        self.assertEqual(result, [2, 3, 4, 4, [5, 2, 3]])

    def test_elementwise_apply_with_no_condition(self):
        @elementwise_apply()
        def add_one(x):
            return x + 1

        data = [1, 2, 3, 4, [5, 1, 2]]
        result = add_one(data)
        self.assertEqual(result, [2, 3, 4, 5, [6, 2, 3]])
    
    def test_empty_data(self):
        @elementwise_apply(func=lambda x: isinstance(x, int))
        def add_one(x):
            return x + 1

        data = []
        result = add_one(data)
        self.assertEqual(result, [])

    def test_nested_dictionaries(self):
        @elementwise_apply(func=lambda x: isinstance(x, int))
        def add_one(x):
            return x + 1

        data = {'a': {'b': {'c': 1}}}
        result = add_one(data)
        self.assertEqual(result, {'a': {'b': {'c': 2}}})
    
    def test_mixed_nested_data(self):
        @elementwise_apply(func=lambda x: isinstance(x, int))
        def add_one(x):
            return x + 1

        data = {'a': [1, 2, {'b': 3, 'c': [4, 5]}]}
        result = add_one(data)
        self.assertEqual(result, {'a': [2, 3, {'b': 4, 'c': [5, 6]}]})

    def test_non_iterable_data(self):
        @elementwise_apply(func=lambda x: isinstance(x, int))
        def add_one(x):
            return x + 1

        data = 42
        result = add_one(data)
        self.assertEqual(result, 43)



    def test_data_with_custom_class(self):
        class CustomClass:
            def __init__(self):
                self.value = 41
        @elementwise_apply(func=lambda x: isinstance(x, CustomClass))
        def process(x):
            x.value = 42
            return x

        data = [CustomClass(), {'a': CustomClass(), 'b': [1, CustomClass()]}]
        result = process(data)
        self.assertEqual(result[0].value, 42)
        self.assertEqual(result[1]['a'].value, 42)
        self.assertEqual(result[1]['b'][1].value, 42)
        
    def test_elementwise_apply_additional_args(self):
        @elementwise_apply(types=int)
        def increment_by_n(x, n):
            return x + n

        data = {"a": [1, 2], "b": {"c": 3}}
        expected_result = {"a": [6, 7], "b": {"c": 8}}
        result = increment_by_n(data, 5)

        self.assertEqual(result, expected_result)
        
    def test_elementwise_apply_additional_kwargs(self):
        @elementwise_apply(types=int)
        def increment_by_n(x, n, calc='mul'):
            if calc == 'add':
                return x + n
            elif calc == 'mul':
                return x * n
            else:
                return x
            

        data = {"a": [1, 2], "b": {"c": 3}}
        expected_result = {"a": [6, 7], "b": {"c": 8}}
        result = increment_by_n(data, 5, calc='add')

        self.assertEqual(result, expected_result)

    def test_exceptions_propagate(self):
        @elementwise_apply(func=lambda x: isinstance(x, int))
        def my_func(x):
            if x == 3:
                raise ValueError("This is a test exception")
            return x * 2

        data = {"a": [1, 3, 5], "b": {"c": 7}}

        with self.assertRaises(ValueError) as context:
            my_func(data)

        self.assertTrue("This is a test exception" in str(context.exception))




if __name__ == '__main__':
    unittest.main()


