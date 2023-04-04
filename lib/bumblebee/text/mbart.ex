defmodule Bumblebee.Text.Mbart do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 50265,
        doc: """
        the vocabulary size of the token embedding. This corresponds to the number of distinct
        tokens that can be represented in model input and output
        """
      ],
      max_positions: [
        default: 1024,
        doc: """
        the vocabulary size of the position embedding. This corresponds to the maximum sequence
        length that this model can process. Typically this is set to a large value just in case,
        such as 512, 1024 or 2048
        """
      ],
      hidden_size: [
        default: 1024,
        doc: "the dimensionality of hidden layers"
      ],
      encoder_num_blocks: [
        default: 12,
        doc: "the number of Transformer blocks in the encoder"
      ],
      decoder_num_blocks: [
        default: 12,
        doc: "the number of Transformer blocks in the decoder"
      ],
      encoder_num_attention_heads: [
        default: 16,
        doc: "the number of attention heads for each attention layer in the encoder"
      ],
      decoder_num_attention_heads: [
        default: 16,
        doc: "the number of attention heads for each attention layer in the decoder"
      ],
      encoder_intermediate_size: [
        default: 4096,
        doc:
          "the dimensionality of the intermediate layer in the transformer feed-forward network (FFN) in the encoder"
      ],
      decoder_intermediate_size: [
        default: 4096,
        doc:
          "the dimensionality of the intermediate layer in the transformer feed-forward network (FFN) in the decoder"
      ],
      scale_embedding: [
        default: false,
        doc: "scale embeddings by dividing by sqrt(hidden_size)"
      ],
      activation: [
        default: :gelu,
        doc: "the activation function"
      ],
      dropout_rate: [
        default: 0.1,
        doc: "the dropout rate for encoder and decoder"
      ],
      attention_dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for attention weights"
      ],
      activation_dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for activations inside fully connected layers"
      ],
      classifier_dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for the classification head"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ]
    ] ++
      Shared.common_options([
        :output_hidden_states,
        :output_attentions,
        :num_labels,
        :id_to_label
      ]) ++
      Shared.token_options(pad_token_id: 1, bos_token_id: 0, eos_token_id: 2) ++
      Shared.generation_options(forced_eos_token_id: 2)

  @moduledoc """
  mBART model family.

  ## Architectures

    * `:base` - plain mBART without any head on top

    * `:for_causal_language_modeling` - mBART with a language modeling
      head. The head returns logits for each token in the original
      sequence

    * `:for_conditional_generation` - mBART with a language modeling
      head. The head returns logits for each token in the original
      sequence

    * `:for_sequence_classification` - mBART with a sequence
      classification head. The head returns logits corresponding to
      possible classes

    * `:for_question_answering` - mBART with a span classification head.
      The head returns logits for the span start and end positions

  ## Inputs

    * `"input_ids"` - `{batch_size, sequence_length}`

      Indices of input sequence tokens in the vocabulary.

    * `"attention_mask"` - `{batch_size, sequence_length}`

      Mask indicating which tokens to attend to. This is used to ignore
      padding tokens, which are added when processing a batch of sequences
      with different length.

    * `"position_ids"` - `{batch_size, sequence_length}`

      Indices of positions of each input sequence tokens in the position
      embeddings.

    * `"attention_head_mask"` - `{encoder_num_blocks, encoder_num_attention_heads}`

      Mask to nullify selected heads of the self-attention blocks in
      the encoder.

    * `"input_embeddings"` - `{batch_size, sequence_length, hidden_size}`

      Embedded representation of `"input_ids"`, which can be specified
      for more control over how `"input_ids"` are embedded than the
      model's internal embedding lookup. If `"input_embeddings"` are present,
      then `"input_ids"` will be ignored.

    * `"decoder_input_ids"` - `{batch_size, target_sequence_length}`

      Indices of decoder input sequence tokens in the vocabulary. If not
      present and `"input_ids"` is, it will be generated by shifting
      each token in `"input_ids"` to the right once.

    * `"decoder_attention_mask"` - `{batch_size, target_sequence_length}`

      Mask indicating which decoder tokens to attend to. This is used
      to ignore padding tokens, which are added when processing a batch
      of sequences with different length.

    * `"decoder_position_ids"` - `{batch_size, target_sequence_length}`

      Indices of positions of each decoder input sequence tokens in
      the position embeddings.

    * `"decoder_attention_head_mask"` - `{decoder_num_blocks, decoder_num_attention_heads}`

      Mask to nullify selected heads of the self-attention blocks in
      the decoder.

    * `"decoder_input_embeddings"` - `{batch_size, sequence_length, hidden_size}`

      Embedded representation of `"decoder_input_ids"`, which can be
      specified for more control over how `"decoder_input_ids"` are
      embedded than the model's internal embedding lookup. If
      `"decoder_input_embeddings"` are present, then `"decoder_input_ids"`
      will be ignored.

    * `"encoder_hidden_state"` - `{batch_size, sequence_length, hidden_size}`

      Last hidden state output from the encoder. This hidden state is
      used in cross-attention blocks in the decoder. If specified, the
      model will skip the encoding process and use this value directly
      for cross-attentions in the decoder.

    * `"cross_attention_head_mask"` - `{decoder_num_blocks, decoder_num_attention_heads}`

      Mask to nullify selected heads of the cross-attention blocks in
      the decoder with shape.

    * `"cache"`

      A container with cached layer results used to speed up sequential
      decoding (autoregression). With cache, certain hidden states are
      taken from the cache, rather than recomputed on every decoding
      pass. The cache should be treated as opaque and initialized with
      `Bumblebee.Text.Generation.init_cache/4`.

  ### Exceptions

  The `:for_causal_language_modeling` model is just the decoder part and
  accepts the following inputs instead: `"input_ids"`, `"attention_mask"`,
  `"position_ids"`, `"attention_head_mask"`, `"input_embeddings"`, `"encoder_hidden_state"`,
  `"encoder_attention_mask"`, `"cross_attention_head_mask"`, `"cache"`.

  ## Configuration

  #{Shared.options_doc(options)}
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
      :for_causal_language_modeling,
      :for_conditional_generation,
      :for_sequence_classification,
      :for_question_answering
    ]

  @impl true
  def config(spec, opts \\ []) do
    spec
    |> Shared.put_config_attrs(opts)
    |> Shared.validate_label_options()
  end

  @impl true
  def input_template(_spec) do
    %{
      "input_ids" => Nx.template({1, 1}, :s64)
    }
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = spec) do
    inputs = encoder_decoder_inputs(spec)

    inputs
    |> core(spec)
    |> Layers.output()
  end

  def model(%__MODULE__{architecture: :for_conditional_generation} = spec) do
    inputs = encoder_decoder_inputs(spec)
    outputs = core(inputs, spec)

    logits =
      outputs.hidden_state
      |> language_modeling_head(spec, name: "language_modeling_head")
      |> Axon.bias(name: "language_modeling_head.logits_bias", bias_initializer: :zeros)

    Layers.output(%{
      logits: logits,
      decoder_hidden_states: outputs.decoder_hidden_states,
      decoder_attentions: outputs.decoder_attentions,
      cross_attentions: outputs.cross_attentions,
      encoder_hidden_state: outputs.encoder_hidden_state,
      encoder_hidden_states: outputs.encoder_hidden_states,
      encoder_attentions: outputs.encoder_attentions,
      cache: outputs.cache
    })
  end

  def model(%__MODULE__{architecture: :for_sequence_classification} = spec) do
    inputs = encoder_decoder_inputs(spec)
    outputs = core(inputs, spec)

    sentence_representation =
      Axon.layer(
        fn input_ids, hidden_state, _opts ->
          eos_mask = Nx.equal(input_ids, spec.eos_token_id)
          eos_idx = Nx.argmax(eos_mask, tie_break: :high, axis: 1)
          Bumblebee.Utils.Nx.batched_take(hidden_state, eos_idx)
        end,
        [inputs["input_ids"], outputs.hidden_state]
      )

    logits =
      sentence_representation
      |> Axon.dropout(rate: spec.classifier_dropout_rate)
      |> Axon.dense(spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: "sequence_classification_head.dense"
      )
      |> Axon.activation(:tanh)
      |> Axon.dropout(rate: spec.classifier_dropout_rate)
      |> Axon.dense(spec.num_labels,
        kernel_initializer: kernel_initializer(spec),
        name: "sequence_classification_head.output"
      )

    Layers.output(%{
      logits: logits,
      decoder_hidden_states: outputs.decoder_hidden_states,
      decoder_attentions: outputs.decoder_attentions,
      cross_attentions: outputs.cross_attentions,
      encoder_hidden_state: outputs.encoder_hidden_state,
      encoder_hidden_states: outputs.encoder_hidden_states,
      encoder_attentions: outputs.encoder_attentions
    })
  end

  def model(%__MODULE__{architecture: :for_question_answering} = spec) do
    inputs = encoder_decoder_inputs(spec)
    outputs = core(inputs, spec)

    logits =
      Axon.dense(outputs.hidden_state, 2,
        kernel_initializer: kernel_initializer(spec),
        name: "question_answering_head.output"
      )

    {start_logits, end_logits} = Layers.split_pair(logits)

    Layers.output(%{
      start_logits: start_logits,
      end_logits: end_logits,
      decoder_hidden_states: outputs.decoder_hidden_states,
      decoder_attentions: outputs.decoder_attentions,
      cross_attentions: outputs.cross_attentions,
      encoder_hidden_state: outputs.encoder_hidden_state,
      encoder_hidden_states: outputs.encoder_hidden_states,
      encoder_attentions: outputs.encoder_attentions
    })
  end

  def model(%__MODULE__{architecture: :for_causal_language_modeling} = spec) do
    shape = {nil, nil}
    hidden_shape = {nil, nil, spec.hidden_size}

    decoder_attention_head_mask_shape =
      {spec.decoder_num_blocks, spec.decoder_num_attention_heads}

    inputs =
      Bumblebee.Utils.Model.inputs_to_map([
        Axon.input("input_ids", optional: true, shape: shape),
        Axon.input("attention_mask", optional: true, shape: shape),
        Axon.input("position_ids", optional: true, shape: shape),
        Axon.input("attention_head_mask",
          optional: true,
          shape: decoder_attention_head_mask_shape
        ),
        Axon.input("input_embeddings", optional: true, shape: hidden_shape),
        Axon.input("encoder_hidden_state", optional: true, shape: hidden_shape),
        Axon.input("encoder_attention_mask", optional: true, shape: shape),
        Axon.input("cross_attention_head_mask",
          optional: true,
          shape: decoder_attention_head_mask_shape
        ),
        Axon.input("cache", optional: true)
      ])

    embeddings =
      embedder(inputs["input_ids"], inputs["position_ids"], inputs["input_embeddings"], spec,
        name: "decoder_embedder"
      )

    outputs =
      decoder(
        embeddings,
        inputs["attention_mask"],
        inputs["attention_head_mask"],
        inputs["encoder_hidden_state"],
        inputs["encoder_attention_mask"],
        inputs["cross_attention_head_mask"],
        inputs["cache"],
        spec,
        name: "decoder"
      )

    logits = language_modeling_head(outputs.hidden_state, spec, name: "language_modeling_head")

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions,
      cross_attentions: outputs.cross_attentions,
      cache: outputs.cache
    })
  end

  defp encoder_decoder_inputs(spec) do
    shape = {nil, nil}
    hidden_shape = {nil, nil, spec.hidden_size}

    encoder_attention_head_mask_shape =
      {spec.encoder_num_blocks, spec.encoder_num_attention_heads}

    decoder_attention_head_mask_shape =
      {spec.decoder_num_blocks, spec.decoder_num_attention_heads}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("input_ids", optional: true, shape: shape),
      Axon.input("attention_mask", optional: true, shape: shape),
      Axon.input("position_ids", optional: true, shape: shape),
      Axon.input("attention_head_mask", optional: true, shape: encoder_attention_head_mask_shape),
      Axon.input("input_embeddings", optional: true, shape: hidden_shape),
      Axon.input("decoder_input_ids", optional: true, shape: shape),
      Axon.input("decoder_attention_mask", optional: true, shape: shape),
      Axon.input("decoder_position_ids", optional: true, shape: shape),
      Axon.input("decoder_attention_head_mask",
        optional: true,
        shape: decoder_attention_head_mask_shape
      ),
      Axon.input("decoder_input_embeddings", optional: true, shape: hidden_shape),
      Axon.input("encoder_hidden_state", optional: true, shape: hidden_shape),
      Axon.input("cross_attention_head_mask",
        optional: true,
        shape: decoder_attention_head_mask_shape
      ),
      Axon.input("cache", optional: true)
    ])
  end

  @impl true
  def init_cache(spec, batch_size, max_length, inputs) do
    encoder_sequence_length =
      if encoder_hidden_state = inputs["encoder_hidden_state"] do
        Nx.axis_size(encoder_hidden_state, 1)
      end

    Layers.Decoder.init_cache(batch_size, max_length,
      hidden_size: spec.hidden_size,
      decoder_num_attention_heads: spec.decoder_num_attention_heads,
      encoder_num_attention_heads: spec.encoder_num_attention_heads,
      decoder_num_blocks: spec.decoder_num_blocks,
      encoder_sequence_length: encoder_sequence_length
    )
  end

  @impl true
  def traverse_cache(_spec, cache, fun) do
    Layers.Decoder.traverse_cache(cache, fun)
  end

  defp core(inputs, spec) do
    encoder_outputs =
      Layers.if_present inputs["encoder_hidden_state"] do
        %{
          hidden_state: inputs["encoder_hidden_state"],
          hidden_states: Layers.none(),
          attentions: Layers.none()
        }
      else
        embeddings =
          embedder(inputs["input_ids"], inputs["position_ids"], inputs["input_embeddings"], spec,
            name: "encoder_embedder"
          )

        embeddings
        |> encoder(inputs["attention_mask"], inputs["attention_head_mask"], spec, name: "encoder")
        |> Map.take([:hidden_state, :hidden_states, :attentions])
      end

    decoder_input_ids =
      Layers.default inputs["decoder_input_ids"] do
        Axon.nx(inputs["input_ids"], fn input_ids ->
          sequence_length = Nx.axis_size(input_ids, 1)

          eos_indices =
            input_ids
            |> Nx.not_equal(spec.pad_token_id)
            |> Nx.sum(axes: [-1])
            |> Nx.subtract(1)
            |> Nx.reshape({:auto, 1})
            |> Nx.as_type({:s, 64})

          # Use the last non-padding token as the decoder start token
          start_ids = Bumblebee.Utils.Nx.batched_take(input_ids, eos_indices)

          if sequence_length == 1 do
            start_ids
          else
            Nx.concatenate([start_ids, input_ids[[0..-1//1, 0..-2//1]]], axis: 1)
          end
        end)
      end

    embeddings =
      embedder(
        decoder_input_ids,
        inputs["decoder_position_ids"],
        inputs["decoder_input_embeddings"],
        spec,
        name: "decoder_embedder"
      )

    decoder_outputs =
      decoder(
        embeddings,
        inputs["decoder_attention_mask"],
        inputs["decoder_attention_head_mask"],
        encoder_outputs.hidden_state,
        inputs["attention_mask"],
        inputs["cross_attention_head_mask"],
        inputs["cache"],
        spec,
        name: "decoder"
      )

    %{
      hidden_state: decoder_outputs.hidden_state,
      decoder_hidden_states: decoder_outputs.hidden_states,
      decoder_attentions: decoder_outputs.attentions,
      cross_attentions: decoder_outputs.cross_attentions,
      cache: decoder_outputs.cache,
      encoder_hidden_state: encoder_outputs.hidden_state,
      encoder_hidden_states: encoder_outputs.hidden_states,
      encoder_attentions: encoder_outputs.attentions
    }
  end

  defp encoder(hidden_state, attention_mask, attention_head_mask, spec, opts) do
    name = opts[:name]

    encoder_outputs =
      Layers.Transformer.blocks(hidden_state,
        attention_mask: attention_mask,
        attention_head_mask: attention_head_mask,
        num_blocks: spec.encoder_num_blocks,
        num_attention_heads: spec.encoder_num_attention_heads,
        hidden_size: spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        dropout_rate: spec.dropout_rate,
        attention_dropout_rate: spec.attention_dropout_rate,
        layer_norm: [
          epsilon: 1.0e-5
        ],
        norm_placement: :first,
        ffn: [
          intermediate_size: spec.encoder_intermediate_size,
          activation: spec.activation
        ],
        output_hidden_states: spec.output_hidden_states,
        output_attentions: spec.output_attentions,
        name: join(name, "blocks")
      )

    hidden_state = Axon.layer_norm(encoder_outputs.hidden_state, name: join(name, "norm"))

    %{
      hidden_state: hidden_state,
      hidden_states: Layers.replace(encoder_outputs.hidden_states, -1, hidden_state),
      attentions: encoder_outputs.attentions
    }
  end

  defp embedder(input_ids, position_ids, input_embeddings, spec, opts) do
    name = opts[:name]

    input_embeddings =
      Layers.default input_embeddings do
        token_embedding(input_ids, spec, name: join(name, "token_embedding"))
      end

    position_ids =
      Layers.default position_ids do
        Layers.default_position_ids(input_embeddings)
      end

    position_embeddings =
      position_embedding(position_ids, spec, name: join(name, "position_embedding"))

    Axon.add([input_embeddings, position_embeddings])
    |> Axon.layer_norm(epsilon: 1.0e-5, name: join(name, "norm"))
    |> Axon.dropout(rate: spec.dropout_rate)
  end

  defp token_embedding(input_ids, spec, opts) do
    name = opts[:name]

    input_embeddings =
      Axon.embedding(input_ids, spec.vocab_size, spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: name
      )

    if spec.scale_embedding do
      Axon.nx(input_embeddings, fn x -> Nx.multiply(x, Nx.sqrt(spec.hidden_size)) end)
    else
      input_embeddings
    end
  end

  defp position_embedding(position_ids, spec, opts) do
    name = opts[:name]

    # For mBART we need to offset the embeddings
    offset = 2

    position_ids
    |> Axon.add(Axon.constant(Nx.tensor(offset)))
    |> Axon.embedding(spec.max_positions + offset, spec.hidden_size, name: name)
  end

  defp decoder(
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

    decoder_outputs =
      Layers.Transformer.blocks(hidden_state,
        attention_mask: attention_mask,
        attention_head_mask: attention_head_mask,
        cross_hidden_state: encoder_hidden_state,
        cross_attention_mask: encoder_attention_mask,
        cross_attention_head_mask: cross_attention_head_mask,
        cache: cache,
        causal?: true,
        num_blocks: spec.decoder_num_blocks,
        num_attention_heads: spec.decoder_num_attention_heads,
        hidden_size: spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        dropout_rate: spec.dropout_rate,
        attention_dropout_rate: spec.attention_dropout_rate,
        layer_norm: [
          epsilon: 1.0e-5
        ],
        norm_placement: :first,
        ffn: [
          intermediate_size: spec.decoder_intermediate_size,
          activation: spec.activation
        ],
        output_hidden_states: spec.output_hidden_states,
        output_attentions: spec.output_attentions,
        name: join(name, "blocks")
      )

    hidden_state = Axon.layer_norm(decoder_outputs.hidden_state, name: join(name, "norm"))

    %{
      cache: decoder_outputs.cache,
      hidden_state: hidden_state,
      hidden_states: Layers.replace(decoder_outputs.hidden_states, -1, hidden_state),
      attentions: decoder_outputs.attentions,
      cross_attentions: decoder_outputs.cross_attentions
    }
  end

  defp language_modeling_head(hidden_state, spec, opts) do
    name = opts[:name]

    # TODO: Tie lm-head to word embedding as a spec option
    Layers.dense_transposed(hidden_state, spec.vocab_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "output")
    )
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
          hidden_size: {"d_model", number()},
          encoder_num_blocks: {"encoder_layers", number()},
          decoder_num_blocks: {"decoder_layers", number()},
          encoder_num_attention_heads: {"encoder_attention_heads", number()},
          decoder_num_attention_heads: {"decoder_attention_heads", number()},
          encoder_intermediate_size: {"encoder_ffn_dim", number()},
          decoder_intermediate_size: {"decoder_ffn_dim", number()},
          scale_embedding: {"scale_embedding", boolean()},
          activation: {"activation_function", atom()},
          dropout_rate: {"dropout", number()},
          attention_dropout_rate: {"attention_dropout", number()},
          activation_dropout_rate: {"activation_dropout", number()},
          classifier_dropout_rate: {"classifier_dropout", number()},
          initializer_scale: {"init_std", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        "encoder_embedder.token_embedding" => "model.encoder.embed_tokens",
        "encoder_embedder.position_embedding" => "model.encoder.embed_positions",
        "encoder_embedder.norm" => "model.encoder.layernorm_embedding",
        "encoder.blocks.{n}.self_attention.query" => "model.encoder.layers.{n}.self_attn.q_proj",
        "encoder.blocks.{n}.self_attention.key" => "model.encoder.layers.{n}.self_attn.k_proj",
        "encoder.blocks.{n}.self_attention.value" => "model.encoder.layers.{n}.self_attn.v_proj",
        "encoder.blocks.{n}.self_attention.output" =>
          "model.encoder.layers.{n}.self_attn.out_proj",
        "encoder.blocks.{n}.self_attention_norm" =>
          "model.encoder.layers.{n}.self_attn_layer_norm",
        "encoder.blocks.{n}.ffn.intermediate" => "model.encoder.layers.{n}.fc1",
        "encoder.blocks.{n}.ffn.output" => "model.encoder.layers.{n}.fc2",
        "encoder.blocks.{n}.output_norm" => "model.encoder.layers.{n}.final_layer_norm",
        "encoder.norm" => "model.encoder.layer_norm",
        "decoder_embedder.token_embedding" => "model.decoder.embed_tokens",
        "decoder_embedder.position_embedding" => "model.decoder.embed_positions",
        "decoder_embedder.norm" => "model.decoder.layernorm_embedding",
        "decoder.blocks.{n}.self_attention.query" => "model.decoder.layers.{n}.self_attn.q_proj",
        "decoder.blocks.{n}.self_attention.key" => "model.decoder.layers.{n}.self_attn.k_proj",
        "decoder.blocks.{n}.self_attention.value" => "model.decoder.layers.{n}.self_attn.v_proj",
        "decoder.blocks.{n}.self_attention.output" =>
          "model.decoder.layers.{n}.self_attn.out_proj",
        "decoder.blocks.{n}.self_attention_norm" =>
          "model.decoder.layers.{n}.self_attn_layer_norm",
        "decoder.blocks.{n}.cross_attention.query" =>
          "model.decoder.layers.{n}.encoder_attn.q_proj",
        "decoder.blocks.{n}.cross_attention.key" =>
          "model.decoder.layers.{n}.encoder_attn.k_proj",
        "decoder.blocks.{n}.cross_attention.value" =>
          "model.decoder.layers.{n}.encoder_attn.v_proj",
        "decoder.blocks.{n}.cross_attention.output" =>
          "model.decoder.layers.{n}.encoder_attn.out_proj",
        "decoder.blocks.{n}.cross_attention_norm" =>
          "model.decoder.layers.{n}.encoder_attn_layer_norm",
        "decoder.blocks.{n}.ffn.intermediate" => "model.decoder.layers.{n}.fc1",
        "decoder.blocks.{n}.ffn.output" => "model.decoder.layers.{n}.fc2",
        "decoder.blocks.{n}.output_norm" => "model.decoder.layers.{n}.final_layer_norm",
        "decoder.norm" => "model.decoder.layer_norm",
        "language_modeling_head.output" => "model.shared",
        "language_modeling_head.logits_bias" => %{
          "bias" => {[{"model", "final_logits_bias"}], fn [value] -> Nx.squeeze(value) end}
        },
        "sequence_classification_head.dense" => "classification_head.dense",
        "sequence_classification_head.output" => "classification_head.out_proj",
        "question_answering_head.output" => "qa_outputs"
      }
    end
  end
end
