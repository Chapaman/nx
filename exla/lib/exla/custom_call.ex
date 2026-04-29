defprotocol EXLA.CustomCall do
  @moduledoc """
  Extension point for lowering selected `Nx.block/4` tags to **XLA custom calls**
  (`stablehlo.custom_call` in MLIR), the same style as helpers on
  `EXLA.MLIR.Value` such as `qr/3` and `eigh/3`.

  Other blocks (for example gather-based `take` or FFT) are lowered inline in
  `EXLA.Defn` and do not use this protocol.

  ## When `EXLA.Defn` calls it

  During compilation with `compiler: EXLA`, when the builder is an MLIR
  `EXLA.MLIR.Function`, each `Nx.block(tag, inputs, outputs, fn ... end)` is
  passed here. `EXLA.Defn` invokes:

    * `function_name(tag, outputs_template, input_templates, client)`
    * `config(tag, outputs_template, input_templates, client)`

  If `function_name/4` returns `:skip`, EXLA compiles the block's **default
  callback** (the anonymous function body) instead of emitting a custom call.

  ## `function_name/4` and `config/4` arguments

    * `struct` — the **tag** passed as the first argument to `Nx.block/4`
      (your own `defstruct` or an existing tag such as `%Nx.Block.LinAlg.QR{}`).

    * `out` — the **output template** tuple passed to `Nx.block/4` (expression
      metadata for shapes and types, not runtime tensors).

    * `args` — list of **input templates**, in the same order as `inputs` in
      `Nx.block/4`.

    * `client` — the active `EXLA.Client` (use e.g. `client.platform` to gate
      host-only lowerings).

  ## Return values

    * `function_name/4`:
      * **Success** — return the native custom-call target name.
      * **`:skip`** — this implementation does not apply (unsupported type,
        non-host platform, wrong arity, etc.). The default block implementation
        is used instead.

    * `config/4`:
      * Return a `map()` to be encoded as `backend_config`.
      * Return `nil` to omit `backend_config`.

  ## Dispatch

  The protocol uses `@fallback_to_any true`. Built-in lowerings for known tags
  live in `defimpl EXLA.CustomCall, for: Any`. Your application or dependency can
  add `defimpl EXLA.CustomCall, for: YourStruct`; that implementation is chosen
  whenever the block tag is `%YourStruct{}`, instead of the `Any` fallback.

  ## Native handlers

  Emitting a custom call in MLIR is only half of the story: the **target name**
  must be registered with XLA on the relevant platform (typically via a native
  library loaded into the process). That registration is **not** configured
  through `config :exla, ...`; you load or link the native code by the same
  means you would for any other NIF-backed extension.

  ## Example

      defmodule MyApp.CustomQrTag do
        defstruct []
      end

      defimpl EXLA.CustomCall, for: MyApp.CustomQrTag do
        def function_name(_tag, {%{type: {kind, size}}, _r_expr}, [_input], %{platform: :host})
            when kind != :c and kind in [:f, :bf] and size in [16, 32, 64] do
          "my_custom_qr_target"
        end

        def function_name(_, _, _, _), do: :skip

        def config(_, _, _, _), do: nil
      end

  Then use `Nx.block(%MyApp.CustomQrTag{}, ...)` inside a `defn` compiled with
  `compiler: EXLA`.
  """

  @fallback_to_any true

  @doc """
  Returns the custom-call target name or `:skip`.
  """
  def function_name(struct, out, args, client)

  @doc """
  Returns a map encoded into `backend_config`, or `nil`.
  """
  def config(struct, out, args, client)
end

defmodule EXLA.CustomCall.Builtins do
  @moduledoc false

  @doc """
  Host CPU `stablehlo.custom_call` target for `Nx.LinAlg.qr/2`, or `:skip`.

  `operand_type` is the input matrix element type; `q_output_type` is the
  element type of the `Q` factor from the block output template.
  """
  def qr_cpu_target(operand_type, q_output_type) do
    case {operand_type, q_output_type} do
      {{:f, 32}, {:f, 32}} -> "qr_cpu_custom_call_f32"
      {{:f, 64}, {:f, 64}} -> "qr_cpu_custom_call_f64"
      {{:f, 16}, {:f, 16}} -> "qr_cpu_custom_call_f16"
      {{:bf, 16}, {:bf, 16}} -> "qr_cpu_custom_call_bf16"
      {{:s, 8}, {:f, 32}} -> "qr_cpu_custom_call_s8"
      {{:s, 16}, {:f, 32}} -> "qr_cpu_custom_call_s16"
      {{:s, 32}, {:f, 32}} -> "qr_cpu_custom_call_s32"
      {{:s, 64}, {:f, 32}} -> "qr_cpu_custom_call_s64"
      {{:u, 8}, {:f, 32}} -> "qr_cpu_custom_call_u8"
      {{:u, 16}, {:f, 32}} -> "qr_cpu_custom_call_u16"
      {{:u, 32}, {:f, 32}} -> "qr_cpu_custom_call_u32"
      {{:u, 64}, {:f, 32}} -> "qr_cpu_custom_call_u64"
      _ -> :skip
    end
  end

  @doc """
  Host CPU `stablehlo.custom_call` target for `Nx.LinAlg.eigh/2`, or `:skip`.

  `operand_type` is the input matrix element type; `computation_type` is the
  floating type used for eigenvalues and eigenvectors (same rule as
  `Nx.Type.merge(Nx.Type.to_floating(evec_type), {:f, 32})` in the protocol).
  Integer operands are promoted to that float inside the native handler.
  """
  def eigh_cpu_target(operand_type, computation_type) do
    case {operand_type, computation_type} do
      {{:f, 32}, {:f, 32}} -> "eigh_cpu_custom_call_f32"
      {{:f, 64}, {:f, 64}} -> "eigh_cpu_custom_call_f64"
      {{:s, 8}, {:f, 32}} -> "eigh_cpu_custom_call_s8"
      {{:s, 16}, {:f, 32}} -> "eigh_cpu_custom_call_s16"
      {{:s, 32}, {:f, 32}} -> "eigh_cpu_custom_call_s32"
      {{:s, 64}, {:f, 32}} -> "eigh_cpu_custom_call_s64"
      {{:u, 8}, {:f, 32}} -> "eigh_cpu_custom_call_u8"
      {{:u, 16}, {:f, 32}} -> "eigh_cpu_custom_call_u16"
      {{:u, 32}, {:f, 32}} -> "eigh_cpu_custom_call_u32"
      {{:u, 64}, {:f, 32}} -> "eigh_cpu_custom_call_u64"
      _ -> :skip
    end
  end
end

# Default EXLA lowerings for **C-backed custom_call** `Nx.block/4` tags live
# in this `defimpl ..., for: Any` module. With `@fallback_to_any true` on the
# protocol, applications and libraries can define their own
# `defimpl EXLA.CustomCall, for: SomeStruct` — protocol dispatch uses that
# implementation instead of this fallback when the block tag matches (you can
# also target a built-in struct such as `Nx.Block...` from your app if needed).
#
defimpl EXLA.CustomCall, for: Any do
  @moduledoc false

  def function_name(
        %Nx.Block.LinAlg.QR{},
        {%{type: q_type}, _r_expr},
        [%{type: in_type} | _],
        %{platform: :host}
      )
      when elem(q_type, 0) != :c and elem(in_type, 0) != :c do
    EXLA.CustomCall.Builtins.qr_cpu_target(in_type, q_type)
  end

  def function_name(
        %Nx.Block.LinAlg.Eigh{},
        {%{type: eval_type}, %{type: evec_type}},
        [%{type: in_type} | _],
        %{platform: :host}
      )
      when elem(eval_type, 0) != :c and elem(evec_type, 0) != :c and
             elem(in_type, 0) != :c do
    computation_type =
      Nx.Type.merge(Nx.Type.to_floating(evec_type), {:f, 32})

    EXLA.CustomCall.Builtins.eigh_cpu_target(in_type, computation_type)
  end

  def function_name(_, _, _, _), do: :skip

  def config(_, _, _, _), do: nil
end
