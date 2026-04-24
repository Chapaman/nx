defprotocol EXLA.CustomCall do
  @moduledoc """
  Protocol used by `EXLA.Defn` to lower specific `Nx.block/4` tags that are
  implemented as **XLA/StableHLO custom calls into native (C/C++) code** —
  the same pipeline as `EXLA.MLIR.Value` helpers such as `qr/3` and `eigh/3`.

  Other blocks (for example gather-based take or plain StableHLO FFT) stay
  inlined in `EXLA.Defn` so this protocol stays focused on what those paths
  share: `stablehlo.custom_call` plus registration of the callee.

  Implementations receive the block tag struct, the output template (`out`),
  the already-recursed MLIR `EXLA.MLIR.Value` arguments and the active
  `EXLA.Client`.

  Built-in lowerings for those tags live in a single `defimpl ..., for: Any`
  module (see comment there). Applications and libraries can still supply a
  **more specific** `defimpl EXLA.CustomCall, for: TheirStruct` — Elixir will
  use that instead of the `Any` fallback when the block tag matches.
  """

  @fallback_to_any true

  @doc """
  Returns `true` when EXLA should lower the block natively via `call/4`.

  When it returns `false`, `EXLA.Defn` falls back to compiling the block's
  default callback implementation.
  """
  def apply?(struct, out, args, client)

  @fallback_to_any true

  @doc """
  Lowers the block natively.

  Must return the list of `EXLA.MLIR.Value`s (or a single value) that
  represents the block result, matching the shape of `out`.
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
  alias EXLA.MLIR.Value
  alias EXLA.Defn

  # --- apply?/4 ---

  def apply?(
        %Nx.Block.LinAlg.QR{},
        {%{type: {q_type_kind, _}}, _r},
        _args,
        client
      ) do
    q_type_kind != :c and client.platform == :host
  end

  def apply?(
        %Nx.Block.LinAlg.Eigh{},
        {%{type: {eval_type_kind, _}}, %{type: {evec_type_kind, _}}},
        _args,
        client
      ) do
    eval_type_kind != :c and evec_type_kind != :c and client.platform == :host
  end

  def apply?(_, _, _, _), do: false

  # --- call/4 ---

  def call(%Nx.Block.LinAlg.QR{}, {q_expr, r_expr}, [tensor], _client) do
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
        {eigenvals_expr, eigenvecs_expr},
        [tensor],
        _client
      ) do
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

  def call(struct, _out, _args, _client) do
    raise ArgumentError,
          "EXLA.CustomCall.call/4 is not implemented for #{inspect(struct)}. " <>
            "Did you forget to guard with EXLA.CustomCall.apply?/4?"
  end
end
