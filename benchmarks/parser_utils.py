import argparse


def override_parser(parser):
    """override some arguments for local"""

    # add kv cache dtype: int8
    parser._option_string_actions["--kv-cache-dtype"].choices += ["int8"]
    parser._option_string_actions["--device"].choices += ["gcu"]

    # set disable_async_output_proc default True
    parser._option_string_actions["--disable-async-output-proc"].default = True

    # set "gcu" to device
    parser._option_string_actions["--device"].choices += ["gcu"]
