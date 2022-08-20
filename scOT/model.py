
"""
This file contains scOT.

A lot of this file is taken from the transformers library and changed to our purposes. Huggingface Transformers is licensed under
Apache 2.0 License, see trainer.py for details.

We follow https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/swinv2/configuration_swinv2.py
and https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/swinv2/modeling_swinv2.py#L1129

The class ConvNeXtBlock is taken from the facebookresearch/ConvNeXt repository and is licensed under the MIT License,