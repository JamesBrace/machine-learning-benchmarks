Firefox does not allow a page to close itself without a user interaction. In the context of this benchmark suite, it means tabs stay opened after a benchmark has finished executing. To allow a benchmark to close itself change the following option in about:config:

allow_scripts_to_close_windows to true.

Some benchmarks may run longer than the default allowed time, which triggers the 'Unresponsive script dialog'. Set the following option to increase the allowed time:

max_script_run_time to 600

