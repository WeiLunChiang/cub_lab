import mlflow
import traceback
from langchain.callbacks import get_openai_callback


def mlflow_exception_logger(func):
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            error_message = f"Uncaught exception: {e}\n{traceback.format_exc()}"
            mlflow.set_tag("uncaught_error_message", e)
            mlflow.log_text(error_message, "uncaught_error_log.txt")
            # raise

    return wrapper


def mlflow_openai_callback(func):
    def wrapper(*args, **kwargs):
        with get_openai_callback() as cb:
            func(*args, **kwargs)
            mlflow.log_text(str(cb), "total_cost.txt")
            mlflow.log_metric("total_cost_USD", cb.total_cost)
            mlflow.log_metric("total_tokens", cb.total_tokens)
            mlflow.log_metric("successful_requests", cb.successful_requests)

    return wrapper
