import argparse


def override_parser(parser):
    """override some arguments for local"""

    # add kv cache dtype: int8
    parser._option_string_actions["--kv-cache-dtype"].choices += ["int8"]

    # set disable_async_output_proc default True
    parser._option_string_actions["--disable-async-output-proc"].default = True
