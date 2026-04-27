# Test-only block tag + `EXLA.CustomCall` impl used to emit a StableHLO custom_call
# with `call_target_name` `qr_cpu_custom_call_f32_exla_alias` (registered by
# `priv/test/exla_qr_alias_plugin.so` when built with `MIX_ENV=test`).
defmodule EXLA.Test.QRAliasBlock do
  @moduledoc false
  defstruct []
end

defimpl EXLA.CustomCall, for: EXLA.Test.QRAliasBlock do
  alias EXLA.MLIR.Value
  alias EXLA.Defn

  def call(_, {%{type: {q_kind, _}} = q_expr, r_expr}, [tensor], client)
      when q_kind != :c and client.platform == :host do
    tensor =
      if Defn.op_type(tensor) != q_expr.type do
        Defn.to_type(tensor, q_expr.type)
      else
        tensor
      end

    {q, r} =
      Value.qr_with_call_target(
        tensor,
        Defn.expr_to_typespec(q_expr),
        Defn.expr_to_typespec(r_expr),
        "qr_cpu_custom_call_f32_exla_alias"
      )

    [q, r]
  end

  def call(_, _, _, _), do: :skip
end
