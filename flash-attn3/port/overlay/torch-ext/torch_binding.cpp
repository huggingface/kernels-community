#include "registration.h"

// The ops are registered by flash-attn/flash_api_stable.cpp. This only provides
// the Python module entry point that the Hub loader imports.
REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
