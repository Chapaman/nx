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
  passed here: `EXLA.Defn` invokes `call(tag, outputs_template, lowered_inputs, client)`.

  If `call/4` returns `:skip`, EXLA compiles the block's **default callback**
  (the anonymous function body) instead of emitting a custom call.

  ## `call/4` arguments

    * `struct` — the **tag** passed as the first argument to `Nx.block/4`
      (your own `defstruct` or an existing tag such as `%Nx.Block.LinAlg.QR{}`).

    * `out` — the **output template** tuple passed to `Nx.block/4` (expression
      metadata for shapes and types, not runtime tensors).

    * `args` — list of already-lowered **operands** as `EXLA.MLIR.Value`s, in
      the same order as `inputs` in `Nx.block/4`.

    * `client` — the active `EXLA.Client` (use e.g. `client.platform` to gate
      host-only lowerings).

  ## Return value

    * **Success** — return a list of `EXLA.MLIR.Value` (or a single value) that
      matches the block result shape implied by `out`.

    * **`:skip`** — this implementation does not apply (unsupported type,
      non-host platform, wrong arity, etc.). The default block implementation is
      used instead.

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
        alias EXLA.Defn
        alias EXLA.MLIR.Value

        def call(_tag, {q_expr, r_expr}, [tensor], %{platform: :host}) do
          tensor =
            if Defn.op_type(tensor) != q_expr.type do
              Defn.to_type(tensor, q_expr.type)
            else
              tensor
            end

          {q, r} =
            Value.qr(tensor, Defn.expr_to_typespec(q_expr), Defn.expr_to_typespec(r_expr))

          [q, r]
        end

        def call(_, _, _, _), do: :skip
      end

  Then use `Nx.block(%MyApp.CustomQrTag{}, ...)` inside a `defn` compiled with
  `compiler: EXLA`.
  """

  @fallback_to_any true

  @doc """
  Attempts to lower the block natively.

  Returns a list of `EXLA.MLIR.Value`s (or a single value) that represents the
  block result, matching the shape of `out`.

  Returns `:skip` when this implementation does not apply (wrong types,
  platform, arity, etc.). `EXLA.Defn` then compiles the block's default callback
  instead.
  """
  def call(struct, out, args, client)
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

  alias EXLA.MLIR.Value
  alias EXLA.Defn

  def call(%Nx.Block.LinAlg.QR{}, {%{type: {q_type_kind, _}} = q_expr, r_expr}, [tensor], client)
      when q_type_kind != :c and client.platform == :host do
    tensor =
      if Defn.op_type(tensor) != q_expr.type do
        Defn.to_type(tensor, q_expr.type)
      else
        tensor
      end

    {q, r} = Value.qr(tensor, Defn.expr_to_typespec(q_expr), Defn.expr_to_typespec(r_expr))
    [q, r]
  end

  def call(
        %Nx.Block.LinAlg.Eigh{},
        {%{type: {eval_type_kind, _}} = eigenvals_expr,
         %{type: {evec_type_kind, _}} = eigenvecs_expr},
        [tensor],
        client
      )
      when eval_type_kind != :c and evec_type_kind != :c and client.platform == :host do
    # Eigen only supports f32/f64, so promote to the smallest floating type
    # wide enough to represent the requested output.
    out_type = Nx.Type.merge(Nx.Type.to_floating(eigenvecs_expr.type), {:f, 32})

    tensor =
      if Defn.op_type(tensor) != out_type do
        Defn.to_type(tensor, out_type)
      else
        tensor
      end

    {eigenvals, eigenvecs} =
      Value.eigh(
        tensor,
        Defn.expr_to_typespec(%{eigenvals_expr | type: out_type}),
        Defn.expr_to_typespec(%{eigenvecs_expr | type: out_type})
      )

    [
      Defn.to_type(eigenvals, eigenvals_expr.type),
      Defn.to_type(eigenvecs, eigenvecs_expr.type)
    ]
  end

  def call(%Nx.Block.LinAlg.QR{}, _, _, _), do: :skip
  def call(%Nx.Block.LinAlg.Eigh{}, _, _, _), do: :skip
  def call(_, _, _, _), do: :skip
end
