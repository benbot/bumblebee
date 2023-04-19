defmodule Bumblebee.Text.MarkupLM do
  alias Bumblebee.Shared

  options = [
    vocab_size: [
      default: 30522,
      doc: """
        the vocabulary size of the token embedding. This corresponds to the number of distinct
        tokens that can be represented in model input and output
      """
    ],
    hidden_size: [
      default: 768,
      doc: """
        the size of the hidden layer in the model
      """
    ],
    num_hidden_layers: [
      default: 12,
      doc: """
        the number of hidden layers in the model
      """
    ],
    num_attention_heads: [
      default: 12,
      doc: """
        the number of attention heads in the model
      """
    ],
    intermediate_size: [
      default: 3072,
      doc: """
        the size of the intermediate layer in the model
      """
    ],
    hidden_act: [
      default: :gelu,
      doc: """
        the activation function to use in the model
      """
    ],
    hidden_dropout_prob: [
      default: 0.1,
      doc: """
        the dropout probability for the hidden layers
      """
    ],
    attention_probs_dropout_prob: [
      default: 0.1,
      doc: """
        the dropout probability for the attention layers
      """
    ],
    max_position_embeddings: [
      default: 512,
      doc: """
        the maximum number of positions in the model
      """
    ],
    type_vocab_size: [
      default: 2,
      doc: """
        the number of types in the model
      """
    ],
    initializer_range: [
      default: 0.02,
      doc: """
        the standard deviation of the normal distribution used to initialize the weights
      """
    ],
    layer_norm_eps: [
      default: 1.0e-12,
      doc: """
        the epsilon value to use in the layer normalization layers
      """
    ],
    max_tree_id_unit_embeddings: [
      default: 1024,
      doc: """
        The maximum value that the xpath subscript unit embedding might ever use. Typically set this to something
      """
    ],
    max_xpath_tag_unit_embeddings: [
      default: 1024,
      doc: """
        The maximum value that the xpath subscript unit embedding might ever use. Typically set this to something
      """
    ],
    max_xpath_subs_unit_embeddings: [
      default: 1024,
      doc: """
        The maximum value that the xpath subscript unit embedding might ever use. Typically set this to something
      """
    ],
    tag_pad_id: [
      default: 216,
      doc: """
        The id of the tag pad token for xpath
      """
    ],
    subs_pad_id: [
      default: 216,
      doc: """
        The id of the subscript pad token for xpath
      """
    ],
    max_depth: [
      default: 50,
      doc: """
        The maximum depth of the xpath
      """
    ],
  ] ++
  Shared.common_options([
    :num_labels,
    :id_to_label,
  ])

  @moduledoc """
  MarkupML Model.

  ## Architectures

  ## Inputs

  ### Exceptions

  ## Configuration

  #{Shared.options_doc(options)}

  ## References


  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable
  @behaviour Bumblebee.Text.Generation

  alias Bumblebee.Layers

  @impl true
  def architectures(),
    do: [
      :base,
      :MarkupLMFromPretrained
    ]

  @impl true
  def config(spec, opts \\ []) do
    spec
    |> Shared.put_config_attrs(opts)
    |> Shared.validate_label_options()
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = spec) do
    inputs = inputs(spec)

    inputs
    |> core(spec)
    |> Layers.output()
  end

  @impl true
  def input_template(_spec) do
    %{"input_ids" => Nx.template({1, 1}, :u32)}
  end

  defp inputs(spec, opts \\ []) do
    shape = {nil, nil}

    hidden_shape = Tuple.append(shape, spec.hidden_size)
    attention_mask_shape = {
      spec.num_hidden_layers,
      spec.num_attention_heads
    }

    inputs =
      Bumblebee.Utils.Model.inputs_to_map([
        Axon.input("input_ids", shape: shape),
        Axon.input("attention_mask", shape: attention_mask_shape, optional: true),
        Axon.input("attention_head_mask", shape: attention_mask_shape, optional: true),
        Axon.input("token_type_ids", shape: shape, optional: true),
        Axon.input("position_ids", shape: shape, optional: true)
      ])

    inputs
  end

  defp core(inputs, spec, opts \\ []) do
    embeddings =
      embedder(
        inputs["input_ids"],
        inputs["position_ids"],
        inputs["token_type_ids"],
        spec,
        name: "embedder"
      )

    encoder_outputs =
      encoder(
        embeddings,
        inputs["attention_mask"],
        inputs["attention_head_mask"],
        spec,
        name: "encoder"
      )

    %{
      hidden_state: encoder_outputs.hidden_state,
      hidden_states: encoder_outputs.hidden_states,
      attentions: encoder_outputs.attentions,
      cross_attentions: encoder_outputs.cross_attentions,
      cache: encoder_outputs.cache
    }
  end

  defp encoder(
    hidden_state,
    attention_mask,
    attention_head_mask,
    spec,
    opts \\ []
  ) do
    name = opts[:name]

    Layers.Transformer.blocks(
      hidden_state,
      attention_mask: attention_mask,
      attention_head_mask: attention_head_mask,
      num_blocks: spec.num_hidden_layers,
      num_attention_heads: spec.num_attention_heads,
      hidden_size: spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      dropout_rate: spec.hidden_dropout_prob,
      attention_dropout_rate: spec.attention_probs_dropout_prob,
      layer_norm: [
        epsilon: spec.layer_norm_eps
      ],
      ffn: [
        intermediate_size: spec.intermediate_size,
        activation: spec.hidden_act,
      ],
      output_attentions: true,
      output_hidden_states: true,
    )
  end

  defp embedder(input_ids, position_ids, token_type_ids, spec, opts \\ []) do
    name = opts[:name]

    position_ids =
      Layers.default position_ids do
        Layers.default_position_ids(input_ids)
      end

    token_type_ids =
      Layers.default token_type_ids do
        Layers.default_position_ids(input_ids)
      end

    input_embeddings =
      Axon.embedding(
        input_ids,
        spec.vocab_size,
        spec.hidden_size,
        kernel_initializer: kernel_initializer(spec)
      )

    position_embeddings =
      Axon.embedding(
        position_ids,
        spec.max_position_embeddings,
        spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: "#{name}.position_embeddings"
      )

    token_type_embeddings =
      Axon.embedding(
        token_type_ids,
        spec.type_vocab_size,
        spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: "#{name}.token_type_embeddings"
      )

    Axon.add([input_embeddings, position_embeddings, token_type_embeddings])
    |> Axon.layer_norm(epsilon: spec.layer_norm_eps, name: "#{name}/LayerNorm")
    |> Axon.dropout(rate: spec.hidden_dropout_prob, name: "#{name}/Dropout")
  end
  defp kernel_initializer(spec), do: Axon.Initializers.normal(scale: spec.initializer_range)

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      opts =
        convert!(data,
          vocab_size: {"vocab_size", number()},
          hidden_size: {"hidden_size", number()},
          num_hidden_layers: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          intermediate_size: {"intermediate_size", number()},
          hidden_act: {:hidden_act, atom()},
          hidden_dropout_prob: {"hidden_dropout_prob", number()},
          attention_probs_dropout_prob: {"attention_probs_dropout_prob", number()},
          max_position_embeddings: {"max_position_embeddings", number()},
          type_vocab_size: {"type_vocab_size", number()},
          initializer_range: {"initializer_range", number()},
          layer_norm_eps: {"layer_norm_eps", number()},
          max_tree_id_unit_embeddings: {"max_tree_id_unit_embeddings", number()},
          max_xpath_tag_unit_embeddings: {"max_xpath_tag_unit_embeddings", number()},
          max_xpath_subs_unit_embeddings: {"max_xpath_subs_unit_embeddings", number()},
          tag_pad_id: {"tag_pad_id", number()},
          subs_pad_id: {"subs_pad_id", number()},
          max_depth: {"max_depth", number()}
        )

        @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{

      }
    end
  end
end
