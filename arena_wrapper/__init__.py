import os


class AppConfig:
    unity_executable_path = os.getenv('UNITY_EXE_PATH')
    unity_log_file = os.getenv('UNITY_LOG_PATH')
    host_pipe_file = os.getenv('HOST_PIPE')
    runtime_platform = os.getenv("RUNTIME_PLATFORM")
    debug = True
