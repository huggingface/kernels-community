
#include <torch/all.h>
#include "RMSNorm.hpp"

torch::Tensor _apply_rms_norm(torch::Tensor const &hidden_states, torch::Tensor const &weight,
                  double variance_epsilon) {
	return rms_norm_impl(hidden_states, {hidden_states.size(-1)}, weight, variance_epsilon);
}
