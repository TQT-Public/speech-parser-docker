import torch


def LlamaModel_fast_forward_inference(hidden_states, config):
    # Ensure hidden_states are cast to a valid PyTorch dtype
    if isinstance(config.torch_dtype, str):
        if config.torch_dtype == "float32":
            dtype = torch.float32
        elif config.torch_dtype == "float16":
            dtype = torch.float16
        else:
            raise ValueError(f"Invalid torch_dtype: {config.torch_dtype}")
    else:
        dtype = config.torch_dtype

    # Cast hidden_states to the correct dtype
    hidden_states = hidden_states.to(dtype)

    # Continue with the forward logic...
    # Proceed with the rest of the LlamaModel_fast_forward_inference process
    outputs = LlamaModel_fast_forward_inference(
        hidden_states=hidden_states,
        # ...  # Other parameters remain unchanged
    )
    return outputs


# # C:\Users\WAR\.conda\envs\unsloth_env\Lib\site-packages\unsloth\models\llama.py
# # in - def LlamaModel_fast_forward_inference - line 900 (approximately)

# # Ensure hidden_states are cast to a valid PyTorch dtype
# if isinstance(self.config.torch_dtype, str):
#     if self.config.torch_dtype == 'float32':
#         dtype = torch.float32
#     elif self.config.torch_dtype == 'float16':
#         dtype = torch.float16
#     else:
#         raise ValueError(f"Invalid torch_dtype: {self.config.torch_dtype}")
# else:
#     dtype = self.config.torch_dtype

# # Cast hidden_states to the correct dtype
# hidden_states = hidden_states.to(dtype)
