from .dnn_regressor import (
    DNNRegressor,
    build_initialized_dnn_regressor,
)

from .simple_cnn_regressor import (
    SimpleCNNRegressor,
    build_initialized_simple_cnn_regressor,
)

MODEL_BUILDERS = {
    "dnn": build_initialized_dnn_regressor,
    "dnn_regressor": build_initialized_dnn_regressor,
    "cnn": build_initialized_simple_cnn_regressor,
    "simple_cnn_regressor": build_initialized_simple_cnn_regressor,
}