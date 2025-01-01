from typing import Callable, Dict, Any
import inspect


def get_function_kwargs(func: Callable[..., Any]) -> list[str]:
    # get function parameters
    pars_dict = dict(inspect.signature(func).parameters)
    # extract only keyword arguments
    return [key for key, val in pars_dict.items() if "=" in val.__str__()]


def match_kwargs(func: Callable[..., Any], kwargs) -> Dict[str, Any]:
    func_kwargs = get_function_kwargs(func)
    passed = list(kwargs.keys())
    # match function kwargs with passsed ones
    return {key: kwargs[key] for key in passed if key in func_kwargs}
