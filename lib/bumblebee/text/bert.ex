defmodule Bumblebee.Text.Bert do
  alias Bumblebee.Shared

  """
      Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the MarkupLM model. Defines the different tokens that can be represented by the
            *inputs_ids* passed to the forward method of [`MarkupLMModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed into [`MarkupLMModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        max_tree_id_unit_embeddings (`int`, *optional*, defaults to 1024):
            The maximum value that the tree id unit embedding might ever use. Typically set this to something large
            just in case (e.g., 1024).
        max_xpath_tag_unit_embeddings (`int`, *optional*, defaults to 256):
            The maximum value that the xpath tag unit embedding might ever use. Typically set this to something large
            just in case (e.g., 256).
        max_xpath_subs_unit_embeddings (`int`, *optional*, defaults to 1024):
            The maximum value that the xpath subscript unit embedding might ever use. Typically set this to something
            large just in case (e.g., 1024).
        tag_pad_id (`int`, *optional*, defaults to 216):
            The id of the padding token in the xpath tags.
        subs_pad_id (`int`, *optional*, defaults to 1001):
            The id of the padding token in the xpath subscripts.
        xpath_tag_unit_hidden_size (`int`, *optional*, defaults to 32):
            The hidden size of each tree id unit. One complete tree index will have
            (50*xpath_tag_unit_hidden_size)-dim.
        max_depth (`int`, *optional*, defaults to 50):
            The maximum depth in xpath.

  """

  options =
    [
      vocab_size: [
        default: 30522,
        doc: """
        the vocabulary size of the token embedding. This corresponds to the number of distinct
        tokens that can be represented in model input and output
        """
      ],
      max_positions: [
        default: 512,
        doc: """
        the vocabulary size of the position embedding. This corresponds to the maximum sequence
        length that this model can process. Typically this is set to a large value just in case,
        such as 512, 1024 or 2048
        """
      ],
      max_token_types: [
        default: 2,
        doc: """
        the vocabulary size of the token type embedding (also referred to as segment embedding).
        This corresponds to how many different token groups can be distinguished in the input
        """
      ],
      hidden_size: [
        default: 768,
        doc: "the dimensionality of hidden layers"
      ],
      num_blocks: [
        default: 12,
        doc: "the number of Transformer blocks in the encoder"
      ],
      num_attention_heads: [
        default: 12,
        doc: "the number of attention heads for each attention layer in the encoder"
      ],
      intermediate_size: [
        default: 3072,
        doc:
          "the dimensionality of the intermediate layer in the transformer feed-forward network (FFN) in the encoder"
      ],
      activation: [
        default: :gelu,
        doc: "the activation function"
      ],
      dropout_rate: [
        default: 0.1,
        doc: "the dropout rate for embedding and encoder"
      ],
      attention_dropout_rate: [
        default: 0.1,
        doc: "the dropout rate for attention weights"
      ],
      classifier_dropout_rate: [
        default: nil,
        doc:
          "the dropout rate for the classification head. If not specified, the value of `:dropout_rate` is used instead"
      ],
      layer_norm_epsilon: [
        default: 1.0e-12,
        doc: "the epsilon used by the layer normalization layers"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ]
    ] ++
      Shared.common_options([
        :use_cross_attention,
        :output_hidden_states,
        :output_attentions,
        :num_labels,
        :id_to_label
      ])

  @moduledoc """
  BERT model family.

  ## Architectures

    * `:base` - plain BERT without any head on top

    * `:for_masked_language_modeling` - BERT with a language modeling
      head. The head returns logits for each token in the original
      sequence

    * `:for_sequence_classification` - BERT with a sequence
      classification head. The head returns logits corresponding to
      possible classes

    * `:for_token_classification` - BERT with a token classification
      head. The head returns logits for each token in the original
      sequence

    * `:for_question_answering` - BERT with a span classification head.
      The head returns logits for the span start and end positions

    * `:for_multiple_choice` - BERT with a multiple choice prediction
      head. Each input in the batch consists of several sequences to
      choose from and the model returns logits corresponding to those
      choices

    * `:for_next_sentence_prediction` - BERT with a next sentence
      prediction head. The head returns logits predicting whether the
      second sentence is random or in context

    * `:for_pre_training` - BERT with both MLM and NSP heads as done
      during the pre-training

    * `:for_causal_language_modeling` - BERT working as a decoder with
      a language modeling head. The head returns logits for each token
      in the original sequence

  ## Inputs

    * `"input_ids"` - `{batch_size, sequence_length}`

      Indices of input sequence tokens in the vocabulary.

    * `"attention_mask"` - `{batch_size, sequence_length}`

      Mask indicating which tokens to attend to. This is used to ignore
      padding tokens, which are added when processing a batch of sequences
      with different length.

    * `"token_type_ids"` - `{batch_size, sequence_length}`

      Mask distinguishing groups in the input sequence. This is used
      in when the input sequence is a semantically a pair of sequences.

    * `"position_ids"` - `{batch_size, sequence_length}`

      Indices of positions of each input sequence tokens in the position
      embeddings.

    * `"attention_head_mask"` - `{num_blocks, num_attention_heads}`

      Mask to nullify selected heads of the self-attention blocks in
      the encoder.

  ### Exceptions

  The `:for_multiple_choice` model accepts groups of sequences, so the
  expected sequence shape is `{batch_size, num_choices, sequence_length}`.

  The `:for_causal_language_modeling` model is a decoder and accepts
  the following additional inputs: `"encoder_hidden_state"`,
  `"encoder_attention_mask"`, `"cross_attention_head_mask"`, `"cache"`.

  ## Configuration

  #{Shared.options_doc(options)}

  ## References

    * [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable
  @behaviour Bumblebee.Text.Generation

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @impl true
  def architectures(),
    do: [
      :base,
      :for_masked_language_modeling,
      :for_sequence_classification,
      :for_token_classification,
      :for_question_answering,
      :for_multiple_choice,
      :for_next_sentence_prediction,
      :for_pre_training,
      :for_causal_language_modeling
    ]

  @impl true
  def config(spec, opts \\ []) do
    spec
    |> Shared.put_config_attrs(opts)
    |> Shared.validate_label_options()
  end

  @impl true
  def input_template(%{architecture: :for_multiple_choice}) do
    %{"input_ids" => Nx.template({1, 1, 1}, :u32)}
  end

  def input_template(_spec) do
    %{"input_ids" => Nx.template({1, 1}, :u32)}
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = spec) do
    inputs = inputs(spec)

    inputs
    |> core(spec)
    |> Layers.output()
  end

  def model(%__MODULE__{architecture: :for_masked_language_modeling} = spec) do
    inputs = inputs(spec)
    outputs = core(inputs, spec)

    logits = language_modeling_head(outputs.hidden_state, spec, name: "language_modeling_head")

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  def model(%__MODULE__{architecture: :for_sequence_classification} = spec) do
    inputs = inputs(spec)
    outputs = core(inputs, spec)

    logits =
      outputs.pooled_state
      |> Axon.dropout(
        rate: classifier_dropout_rate(spec),
        name: "sequence_classification_head.dropout"
      )
      |> Axon.dense(spec.num_labels,
        kernel_initializer: kernel_initializer(spec),
        name: "sequence_classification_head.output"
      )

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  def model(%__MODULE__{architecture: :for_token_classification} = spec) do
    inputs = inputs(spec)
    outputs = core(inputs, spec)

    logits =
      outputs.hidden_state
      |> Axon.dropout(
        rate: classifier_dropout_rate(spec),
        name: "token_classification_head.dropout"
      )
      |> Axon.dense(spec.num_labels,
        kernel_initializer: kernel_initializer(spec),
        name: "token_classification_head.output"
      )

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  def model(%__MODULE__{architecture: :for_question_answering} = spec) do
    inputs = inputs(spec)
    outputs = core(inputs, spec)

    logits =
      outputs.hidden_state
      |> Axon.dropout(
        rate: classifier_dropout_rate(spec),
        name: "question_answering_head.dropout"
      )
      |> Axon.dense(2,
        kernel_initializer: kernel_initializer(spec),
        name: "question_answering_head.output"
      )

    {start_logits, end_logits} = Layers.split_pair(logits)

    Layers.output(%{
      start_logits: start_logits,
      end_logits: end_logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  def model(%__MODULE__{architecture: :for_multiple_choice} = spec) do
    inputs = inputs(spec, shape: {nil, nil, nil})

    group_inputs = ["input_ids", "attention_mask", "token_type_ids", "position_ids"]

    flat_inputs =
      Enum.reduce(group_inputs, inputs, fn name, inputs ->
        Map.update!(inputs, name, &Layers.flatten_leading/1)
      end)

    outputs = core(flat_inputs, spec)

    logits =
      outputs.pooled_state
      |> Axon.dropout(rate: classifier_dropout_rate(spec), name: "multiple_choice_head.dropout")
      |> Axon.dense(1,
        kernel_initializer: kernel_initializer(spec),
        name: "multiple_choice_head.output"
      )

    # The final shape depends on the dynamic batch size and number
    # of choices, so we do a reshape based on the input shape
    logits =
      Axon.layer(
        fn logits, input_ids, _opts ->
          num_choices = Nx.axis_size(input_ids, 1)
          Nx.reshape(logits, {:auto, num_choices})
        end,
        [logits, inputs["input_ids"]]
      )

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  def model(%__MODULE__{architecture: :for_next_sentence_prediction} = spec) do
    inputs = inputs(spec)
    outputs = core(inputs, spec)

    logits =
      Axon.dense(outputs.pooled_state, 2,
        kernel_initializer: kernel_initializer(spec),
        name: "next_sentence_prediction_head.output"
      )

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  def model(%__MODULE__{architecture: :for_pre_training} = spec) do
    inputs = inputs(spec)
    outputs = core(inputs, spec)

    lm_logits = language_modeling_head(outputs.hidden_state, spec, name: "language_modeling_head")

    nsp_logits =
      Axon.dense(outputs.pooled_state, 2,
        kernel_initializer: kernel_initializer(spec),
        name: "next_sentence_prediction_head.output"
      )

    Layers.output(%{
      language_modeling_logits: lm_logits,
      next_sentence_prediction_logits: nsp_logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  def model(%__MODULE__{architecture: :for_causal_language_modeling} = spec) do
    inputs = inputs(spec, decoder?: true)
    outputs = core(inputs, spec, decoder?: true)
    logits = language_modeling_head(outputs.hidden_state, spec, name: "language_modeling_head")

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions,
      cross_attentions: outputs.cross_attentions,
      cache: outputs.cache
    })
  end

  @impl true
  def init_cache(spec, batch_size, max_length, inputs) do
    encoder_sequence_length =
      if encoder_hidden_state = inputs["encoder_hidden_state"] do
        Nx.axis_size(encoder_hidden_state, 1)
      end

    Layers.Decoder.init_cache(batch_size, max_length,
      hidden_size: spec.hidden_size,
      decoder_num_attention_heads: spec.num_attention_heads,
      encoder_num_attention_heads: spec.num_attention_heads,
      decoder_num_blocks: spec.num_blocks,
      encoder_sequence_length: encoder_sequence_length
    )
  end

  @impl true
  def traverse_cache(_spec, cache, fun) do
    Layers.Decoder.traverse_cache(cache, fun)
  end

  defp inputs(spec, opts \\ []) do
    shape = Keyword.get(opts, :shape, {nil, nil})
    decoder? = Keyword.get(opts, :decoder?, false)

    hidden_shape = Tuple.append(shape, spec.hidden_size)
    attention_head_mask_shape = {spec.num_blocks, spec.num_attention_heads}

    inputs =
      Bumblebee.Utils.Model.inputs_to_map([
        Axon.input("input_ids", shape: shape),
        Axon.input("attention_mask", optional: true, shape: shape),
        Axon.input("token_type_ids", optional: true, shape: shape),
        Axon.input("position_ids", optional: true, shape: shape),
        Axon.input("attention_head_mask", optional: true, shape: attention_head_mask_shape)
      ])

    extra_decoder_inputs =
      Bumblebee.Utils.Model.inputs_to_map([
        Axon.input("encoder_hidden_state", optional: true, shape: hidden_shape),
        Axon.input("encoder_attention_mask", optional: true, shape: shape),
        Axon.input("cross_attention_head_mask", optional: true, shape: attention_head_mask_shape),
        Axon.input("cache", optional: true)
      ])

    extra_decoder_inputs =
      if decoder? do
        extra_decoder_inputs
      else
        Map.new(extra_decoder_inputs, fn {name, _input} -> {name, Layers.none()} end)
      end

    Map.merge(inputs, extra_decoder_inputs)
  end

  defp core(inputs, spec, opts \\ []) do
    decoder? = Keyword.get(opts, :decoder?, false)

    embeddings =
      embedder(inputs["input_ids"], inputs["position_ids"], inputs["token_type_ids"], spec,
        name: "embedder"
      )

    encoder_outputs =
      encoder(
        embeddings,
        inputs["attention_mask"],
        inputs["attention_head_mask"],
        inputs["encoder_hidden_state"],
        inputs["encoder_attention_mask"],
        inputs["cross_attention_head_mask"],
        inputs["cache"],
        spec,
        decoder?: decoder?,
        name: "encoder"
      )

    pooled_state = pooler(encoder_outputs.hidden_state, spec, name: "pooler")

    %{
      hidden_state: encoder_outputs.hidden_state,
      pooled_state: pooled_state,
      hidden_states: encoder_outputs.hidden_states,
      attentions: encoder_outputs.attentions,
      cross_attentions: encoder_outputs.cross_attentions,
      cache: encoder_outputs.cache
    }
  end

  defp embedder(input_ids, position_ids, token_type_ids, spec, opts) do
    name = opts[:name]

    position_ids =
      Layers.default position_ids do
        Layers.default_position_ids(input_ids)
      end

    token_type_ids =
      Layers.default token_type_ids do
        Layers.default_token_type_ids(input_ids)
      end

    inputs_embeddings =
      Axon.embedding(input_ids, spec.vocab_size, spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "token_embedding")
      )

    position_embeddings =
      Axon.embedding(position_ids, spec.max_positions, spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "position_embedding")
      )

    token_type_embeddings =
      Axon.embedding(token_type_ids, spec.max_token_types, spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "token_type_embedding")
      )

    Axon.add([inputs_embeddings, position_embeddings, token_type_embeddings])
    |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "norm"))
    |> Axon.dropout(rate: spec.dropout_rate, name: join(name, "dropout"))
  end

  defp encoder(
         hidden_state,
         attention_mask,
         attention_head_mask,
         encoder_hidden_state,
         encoder_attention_mask,
         cross_attention_head_mask,
         cache,
         spec,
         opts
       ) do
    name = opts[:name]
    decoder? = opts[:decoder?]

    cross_attention? = decoder? and spec.use_cross_attention

    Layers.Transformer.blocks(
      hidden_state,
      [
        attention_mask: attention_mask,
        attention_head_mask: attention_head_mask,
        cache: cache,
        causal?: decoder?,
        num_blocks: spec.num_blocks,
        num_attention_heads: spec.num_attention_heads,
        hidden_size: spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        dropout_rate: spec.dropout_rate,
        attention_dropout_rate: spec.attention_dropout_rate,
        layer_norm: [
          epsilon: spec.layer_norm_epsilon
        ],
        ffn: [
          intermediate_size: spec.intermediate_size,
          activation: spec.activation
        ],
        output_hidden_states: spec.output_hidden_states,
        output_attentions: spec.output_attentions,
        name: join(name, "blocks")
      ] ++
        if(cross_attention?,
          do: [
            cross_hidden_state: encoder_hidden_state,
            cross_attention_mask: encoder_attention_mask,
            cross_attention_head_mask: cross_attention_head_mask
          ],
          else: []
        )
    )
  end

  defp pooler(hidden_state, spec, opts) do
    name = opts[:name]

    hidden_state
    |> Layers.take_token(index: 0, axis: 1)
    |> Axon.dense(spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "output")
    )
    |> Axon.tanh()
  end

  defp language_modeling_head(hidden_state, spec, opts) do
    name = opts[:name]

    # TODO: use a shared parameter with embeddings.word_embeddings.kernel
    # if spec.tie_word_embeddings is true (relevant for training)

    hidden_state
    |> Axon.dense(spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "dense")
    )
    |> Layers.activation(spec.activation, name: join(name, "activation"))
    |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "norm"))
    # We reuse the kernel of input embeddings and add bias for each token
    |> Layers.dense_transposed(spec.vocab_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "output")
    )
    |> Axon.bias(name: join(name, "bias"))
  end

  defp classifier_dropout_rate(spec) do
    spec.classifier_dropout_rate || spec.dropout_rate
  end

  defp kernel_initializer(spec) do
    Axon.Initializers.normal(scale: spec.initializer_scale)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      opts =
        convert!(data,
          vocab_size: {"vocab_size", number()},
          max_positions: {"max_position_embeddings", number()},
          max_token_types: {"type_vocab_size", number()},
          hidden_size: {"hidden_size", number()},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          intermediate_size: {"intermediate_size", number()},
          activation: {"hidden_act", atom()},
          dropout_rate: {"hidden_dropout_prob", number()},
          attention_dropout_rate: {"attention_probs_dropout_prob", number()},
          classifier_dropout_rate: {"classifier_dropout", optional(number())},
          layer_norm_epsilon: {"layer_norm_eps", number()},
          initializer_scale: {"initializer_range", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        "embedder.token_embedding" => "bert.embeddings.word_embeddings",
        "embedder.position_embedding" => "bert.embeddings.position_embeddings",
        "embedder.token_type_embedding" => "bert.embeddings.token_type_embeddings",
        "embedder.norm" => "bert.embeddings.LayerNorm",
        "encoder.blocks.{n}.self_attention.query" =>
          "bert.encoder.layer.{n}.attention.self.query",
        "encoder.blocks.{n}.self_attention.key" => "bert.encoder.layer.{n}.attention.self.key",
        "encoder.blocks.{n}.self_attention.value" =>
          "bert.encoder.layer.{n}.attention.self.value",
        "encoder.blocks.{n}.self_attention.output" =>
          "bert.encoder.layer.{n}.attention.output.dense",
        "encoder.blocks.{n}.self_attention_norm" =>
          "bert.encoder.layer.{n}.attention.output.LayerNorm",
        "encoder.blocks.{n}.cross_attention.query" =>
          "bert.encoder.layer.{n}.crossattention.self.query",
        "encoder.blocks.{n}.cross_attention.key" =>
          "bert.encoder.layer.{n}.crossattention.self.key",
        "encoder.blocks.{n}.cross_attention.value" =>
          "bert.encoder.layer.{n}.crossattention.self.value",
        "encoder.blocks.{n}.cross_attention.output" =>
          "bert.encoder.layer.{n}.crossattention.output.dense",
        "encoder.blocks.{n}.cross_attention_norm" =>
          "bert.encoder.layer.{n}.crossattention.output.LayerNorm",
        "encoder.blocks.{n}.ffn.intermediate" => "bert.encoder.layer.{n}.intermediate.dense",
        "encoder.blocks.{n}.ffn.output" => "bert.encoder.layer.{n}.output.dense",
        "encoder.blocks.{n}.output_norm" => "bert.encoder.layer.{n}.output.LayerNorm",
        "pooler.output" => "bert.pooler.dense",
        "language_modeling_head.dense" => "cls.predictions.transform.dense",
        "language_modeling_head.norm" => "cls.predictions.transform.LayerNorm",
        "language_modeling_head.output" => "cls.predictions.decoder",
        "language_modeling_head.bias" => "cls.predictions",
        "next_sentence_prediction_head.output" => "cls.seq_relationship",
        "sequence_classification_head.output" => "classifier",
        "token_classification_head.output" => "classifier",
        "multiple_choice_head.output" => "classifier",
        "question_answering_head.output" => "qa_outputs"
      }
    end
  end
end
