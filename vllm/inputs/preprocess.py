# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mini-vLLM: multimodal support removed (text-only)

from typing import Any, cast

from typing_extensions import assert_never

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike

from .data import (
    DecoderOnlyInputs,
    EmbedsInputs,
    EmbedsPrompt,
    EncoderDecoderInputs,
    ExplicitEncoderDecoderPrompt,
    ProcessorInputs,
    PromptType,
    SingletonInputs,
    SingletonPrompt,
    TextPrompt,
    TokenInputs,
    TokensPrompt,
    embeds_inputs,
    token_inputs,
)
from .parse import is_explicit_encoder_decoder_prompt, parse_singleton_prompt

logger = init_logger(__name__)


class InputPreprocessor:
    """mini-vLLM: Simplified input preprocessor for text-only models."""

    def __init__(
        self,
        model_config: ModelConfig,
        tokenizer: TokenizerLike | None,
        **kwargs,  # Accept and ignore multimodal kwargs for compatibility
    ) -> None:
        super().__init__()

        self.model_config = model_config
        self.tokenizer = tokenizer

    def get_tokenizer(self) -> TokenizerLike:
        if self.tokenizer is None:
            raise ValueError(
                "You cannot pass text prompts when `skip_tokenizer_init=True`"
            )

        return self.tokenizer

    def get_bos_token_id(self) -> int | None:
        if self.tokenizer is None:
            logger.warning_once(
                "Using None for BOS token id because tokenizer is not initialized"
            )
            return None

        return self.tokenizer.bos_token_id

    def get_eos_token_id(self) -> int | None:
        if self.tokenizer is None:
            logger.warning_once(
                "Using None for EOS token id because tokenizer is not initialized"
            )
            return None

        return self.tokenizer.eos_token_id

    def get_decoder_start_token_id(self) -> int | None:
        """
        Obtain the decoder start token id employed by an encoder/decoder
        model. Returns None for non-encoder/decoder models or if the
        model config is unavailable.
        """

        if not self.model_config.is_encoder_decoder:
            logger.warning_once(
                "Using None for decoder start token id because "
                "this is not an encoder/decoder model."
            )
            return None

        if self.model_config is None or self.model_config.hf_config is None:
            logger.warning_once(
                "Using None for decoder start token id because "
                "model config is not available."
            )
            return None

        dec_start_token_id = getattr(
            self.model_config.hf_config, "decoder_start_token_id", None
        )
        if dec_start_token_id is None:
            logger.warning_once(
                "Falling back on <BOS> for decoder start token "
                "id because decoder start token id is not "
                "available."
            )
            dec_start_token_id = self.get_bos_token_id()

        return dec_start_token_id

    def _get_default_enc_dec_decoder_prompt(self) -> list[int]:
        """Generate a default decoder prompt for encoder/decoder models."""
        bos_token_id = self.get_bos_token_id()
        assert bos_token_id is not None
        return [bos_token_id]

    def _prepare_decoder_input_ids_for_generation(
        self,
        decoder_input_ids: list[int] | None,
    ) -> list[int]:
        """Prepares `decoder_input_ids` for generation with encoder-decoder models."""
        decoder_start_token_id = self.get_decoder_start_token_id()
        assert decoder_start_token_id is not None

        if decoder_input_ids is None:
            decoder_input_ids = self._get_default_enc_dec_decoder_prompt()

        if (
            len(decoder_input_ids) == 0
            or decoder_input_ids[0] != decoder_start_token_id
        ):
            decoder_input_ids = [decoder_start_token_id] + decoder_input_ids

        return decoder_input_ids

    def _get_tokenization_kw(
        self,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        kwargs = dict[str, Any]()

        if self.model_config.is_encoder_decoder:
            kwargs["add_special_tokens"] = False

        if overrides:
            kwargs.update(overrides)

        return kwargs

    def _tokenize_prompt(
        self,
        prompt: str,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[int]:
        """Apply the model's tokenizer to a text prompt."""
        tokenizer = self.get_tokenizer()
        tokenization_kwargs = self._get_tokenization_kw(tokenization_kwargs)

        encoder_config = self.model_config.encoder_config

        if encoder_config and encoder_config.get("do_lower_case", False):
            prompt = prompt.lower()

        return tokenizer.encode(prompt, **tokenization_kwargs)

    def _process_embeds(
        self,
        parsed_content: EmbedsPrompt,
    ) -> EmbedsInputs:
        if not self.model_config.enable_prompt_embeds:
            raise ValueError(
                "You must set `--enable-prompt-embeds` to input `prompt_embeds`."
            )

        prompt_embeds = parsed_content["prompt_embeds"]

        if prompt_embeds.ndim == 3:
            prompt_embeds = prompt_embeds.squeeze(dim=0)

        if prompt_embeds.ndim != 2:
            raise ValueError("prompt_embeds must be of shape (seq_len, hidden_size).")

        prompt_embeds = prompt_embeds.cpu()

        return embeds_inputs(
            prompt_embeds=prompt_embeds, cache_salt=parsed_content.get("cache_salt")
        )

    def _truncate_inputs(
        self, inputs: list[int], tokenization_kwargs: dict[str, Any] | None = None
    ) -> list[int]:
        if (
            not tokenization_kwargs
            or "truncation" not in tokenization_kwargs
            or self.tokenizer is None
        ):
            return inputs

        max_length = tokenization_kwargs["max_length"]

        if self.tokenizer.truncation_side == "left":
            return inputs[-max_length:]
        else:
            return inputs[:max_length]

    def _process_tokens(
        self,
        parsed_content: TokensPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> TokenInputs:
        prompt_token_ids = self._truncate_inputs(
            parsed_content["prompt_token_ids"], tokenization_kwargs
        )

        # mini-vLLM: multimodal processing removed, text-only
        inputs = token_inputs(prompt_token_ids)

        if cache_salt := parsed_content.get("cache_salt"):
            inputs["cache_salt"] = cache_salt

        return inputs

    def _process_text(
        self,
        parsed_content: TextPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> TokenInputs:
        prompt_text = parsed_content["prompt"]

        # mini-vLLM: multimodal processing removed, text-only
        prompt_token_ids = self._tokenize_prompt(
            prompt_text,
            tokenization_kwargs=tokenization_kwargs,
        )
        inputs = token_inputs(prompt_token_ids)

        if cache_salt := parsed_content.get("cache_salt"):
            inputs["cache_salt"] = cache_salt

        return inputs

    def _prompt_to_llm_inputs(
        self,
        prompt: SingletonPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> SingletonInputs:
        """Extract the singleton inputs from a prompt."""
        parsed = parse_singleton_prompt(prompt)

        if parsed["type"] == "embeds":
            return self._process_embeds(parsed["content"])
        if parsed["type"] == "tokens":
            return self._process_tokens(parsed["content"])
        if parsed["type"] == "text":
            return self._process_text(
                parsed["content"],
                tokenization_kwargs=tokenization_kwargs,
            )
        if parsed["type"] == "str":
            return self._process_text(
                TextPrompt(prompt=parsed["content"]),
                tokenization_kwargs=tokenization_kwargs,
            )

        assert_never(parsed)

    def _build_enc_dec_llm_inputs(
        self,
        encoder_inputs: SingletonInputs,
        decoder_inputs: SingletonInputs | None,
    ) -> EncoderDecoderInputs:
        if (
            encoder_inputs["type"] == "embeds"
            or decoder_inputs
            and decoder_inputs["type"] == "embeds"
        ):
            raise ValueError(
                "Embedding inputs are not supported for encoder-decoder models"
            )

        encoder_inputs = cast(TokenInputs, encoder_inputs)
        decoder_inputs = cast(TokenInputs | None, decoder_inputs)

        if decoder_inputs is None:
            dec_token_ids = self._prepare_decoder_input_ids_for_generation(None)
            decoder_inputs = token_inputs(dec_token_ids)
        else:
            dec_token_ids = self._prepare_decoder_input_ids_for_generation(
                decoder_inputs["prompt_token_ids"]
            )
            decoder_inputs["prompt_token_ids"] = dec_token_ids

        return EncoderDecoderInputs(
            encoder=encoder_inputs,
            decoder=decoder_inputs,
        )

    def _process_encoder_decoder_prompt(
        self,
        prompt: PromptType,
        tokenization_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> EncoderDecoderInputs:
        """Process an input prompt into EncoderDecoderInputs."""
        encoder_inputs: SingletonInputs
        decoder_inputs: SingletonInputs | None
        if is_explicit_encoder_decoder_prompt(prompt):
            prompt_ = cast(ExplicitEncoderDecoderPrompt, prompt)
            encoder_inputs = self._prompt_to_llm_inputs(
                prompt_["encoder_prompt"],
                tokenization_kwargs=tokenization_kwargs,
            )
            if (decoder_input := prompt_["decoder_prompt"]) is None:
                decoder_inputs = None
            else:
                decoder_inputs = self._prompt_to_llm_inputs(
                    decoder_input, tokenization_kwargs=tokenization_kwargs
                )
        else:
            encoder_inputs = self._prompt_to_llm_inputs(
                cast(SingletonPrompt, prompt),
                tokenization_kwargs=tokenization_kwargs,
            )
            decoder_inputs = None

        return self._build_enc_dec_llm_inputs(encoder_inputs, decoder_inputs)

    def _build_decoder_only_llm_inputs(
        self,
        prompt_inputs: DecoderOnlyInputs,
    ) -> DecoderOnlyInputs:
        return prompt_inputs

    def _process_decoder_only_prompt(
        self,
        prompt: SingletonPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> DecoderOnlyInputs:
        """Process an input prompt into DecoderOnlyInputs."""
        prompt_comps = self._prompt_to_llm_inputs(
            prompt,
            tokenization_kwargs=tokenization_kwargs,
        )

        return self._build_decoder_only_llm_inputs(prompt_comps)

    def _preprocess(
        self,
        prompt: PromptType,
        tokenization_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> ProcessorInputs:
        if self.model_config.is_encoder_decoder:
            return self._process_encoder_decoder_prompt(
                prompt,
                tokenization_kwargs,
            )

        if is_explicit_encoder_decoder_prompt(prompt):
            raise ValueError(
                "Cannot pass encoder-decoder prompt to decoder-only models"
            )

        return self._process_decoder_only_prompt(
            cast(SingletonPrompt, prompt),
            tokenization_kwargs=tokenization_kwargs,
        )

    def preprocess(
        self,
        prompt: PromptType,
        tokenization_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> ProcessorInputs:
        """Preprocess the input prompt."""
        return self._preprocess(
            prompt,
            tokenization_kwargs,
        )

    # mini-vLLM: multimodal cache methods removed (no-op stubs for compatibility)
    def stat_mm_cache(self) -> None:
        return None

    def clear_mm_cache(self) -> None:
        pass
