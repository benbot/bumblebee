defmodule Bumblebee.Text.MarkupLMTokenizer do
  import Bumblebee.Shared

  tokenizer_impl(
    special_tokens: %{
      bos: "<s>",
      eos: "</s>",
      sep: "</s>",
      cls: "<s>",
      unk: "<unk>",
      pad: "<pad>",
      mask: "<mask>",
    }
  )
end
