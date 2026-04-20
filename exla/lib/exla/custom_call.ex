defprotocol EXLA.CustomCall do
  @moduledoc """
  Protocol used by `EXLA.Defn` to lower specific `Nx.block/4` tags natively
  instead of compiling the fallback callback.

  Implementations receive the block tag struct, the output template (`out`),
  the already-recursed MLIR `EXLA.MLIR.Value` arguments and the active
  `EXLA.Client`.
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

  def apply?(%Nx.Block.Take{}, _out, _args, _client), do: true
  def apply?(%Nx.Block.TopK{}, _out, _args, _client), do: true
  def apply?(%Nx.Block.FFT2{}, _out, _args, _client), do: true
  def apply?(%Nx.Block.IFFT2{}, _out, _args, _client), do: true
  def apply?(%Nx.Block.RFFT{}, _out, _args, _client), do: true
  def apply?(%Nx.Block.IRFFT{}, _out, _args, _client), do: true

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

  def call(%Nx.Block.Take{axis: axis}, expr, [tensor, indices], _client) do
    tensor_shape = Defn.op_shape(tensor)
    tensor_rank = tuple_size(tensor_shape)
    indices_rank = indices |> Defn.op_shape() |> tuple_size()
    result_rank = tensor_rank - 1 + indices_rank

    index_vector_dim = indices_rank
    slice_sizes = tensor_shape |> put_elem(axis, 1) |> Tuple.to_list()

    {left, right} = result_rank |> Defn.axes_for_rank() |> Enum.split(axis)
    offset_dims = left ++ Enum.drop(right, indices_rank)

    collapsed_slice_dims = [axis]
    start_index_map = [axis]

    Value.gather(
      tensor,
      indices,
      index_vector_dim,
      slice_sizes,
      offset_dims,
      collapsed_slice_dims,
      start_index_map,
      Defn.expr_to_typespec(expr)
    )
  end

  def call(%Nx.Block.TopK{k: k}, {values, idx}, [tensor], _client) do
    typespecs = [Defn.expr_to_typespec(values), Defn.expr_to_typespec(idx)]
    Value.top_k(tensor, k, typespecs)
  end

  def call(%Nx.Block.FFT2{} = struct, expr, [tensor], _client) do
    Defn.fft2(&Value.fft(&1, :fft, &2, &3), [tensor, fft2_opts(struct)], expr)
  end

  def call(%Nx.Block.IFFT2{} = struct, expr, [tensor], _client) do
    Defn.fft2(&Value.fft(&1, :ifft, &2, &3), [tensor, fft2_opts(struct)], expr)
  end

  def call(%Nx.Block.RFFT{} = struct, expr, [tensor], _client) do
    # expr.type is complex; input tensor is real.
    input_type = Nx.Type.to_real(expr.type)

    Defn.fft(
      &Value.fft(&1, :rfft, &2, &3),
      input_type,
      expr.type,
      [tensor, fft_opts(struct)],
      expr
    )
  end

  def call(%Nx.Block.IRFFT{} = struct, expr, [tensor], _client) do
    # expr.type is real; input tensor is complex. The expected input length is
    # div(n, 2) + 1 (pad_n) while the output length is n (fft_n).
    n = struct.length
    input_type = Nx.Type.to_complex(expr.type)

    Defn.fft(
      &Value.fft(&1, :irfft, &2, &3),
      input_type,
      expr.type,
      div(n, 2) + 1,
      [tensor, fft_opts(struct)],
      expr
    )
  end

  def call(struct, _out, _args, _client) do
    raise ArgumentError,
          "EXLA.CustomCall.call/4 is not implemented for #{inspect(struct)}. " <>
            "Did you forget to guard with EXLA.CustomCall.apply?/4?"
  end

  defp fft_opts(%{length: length, axis: axis, eps: eps}) do
    opts = [length: length, axis: axis]
    if eps, do: Keyword.put(opts, :eps, eps), else: opts
  end

  defp fft2_opts(%{lengths: lengths, axes: axes, eps: eps}) do
    opts = [lengths: lengths, axes: axes]
    if eps, do: Keyword.put(opts, :eps, eps), else: opts
  end
end
