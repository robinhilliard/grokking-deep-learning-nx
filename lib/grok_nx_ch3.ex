defmodule GrokNxCh3 do
  @moduledoc """
  Grokking Deep Learning Examples ported to NX
  Chapter 3
  """
  
  import Nx.Defn

  # Trivial, but idiomatic to Nx for this sort of function
  # to be a defn not a def.
  defn neural_network(input, weights) do
    Nx.dot(input, weights)
  end
  
  def example_1() do
    # This is the first example ported, based on the NumPy
    # example towards the end of Chapter 3.
    
    weights = Nx.tensor([0.1, 0.2, 0.0])
    
    # toes = number of pitches/bowls
    # wlrec = win/loss record
    # nfans = number of fans
    # See below for why this was combined into a single tensor.
    toes_wlrec_nfans = Nx.tensor([
        [8.5, 9.5, 9.9, 9.0],
        [0.65, 0.8, 0.8, 0.9],
        [1.2, 1.3, 0.5, 1.0]
      ])
    
    # Building the input tensor out of separate toes/wlrec/nfans
    # tensors was really inelegant - you couldn't access raw floats
    # easily inside the tensor. For this reason I combined the three
    # separate tensors into a single test/training tensor and sliced
    # the input (first column) out of the combined tensor. You need
    # to transpose the input column to a row before passing to
    # the network.
    input = toes_wlrec_nfans
            |> Nx.slice_axis(0, 1, 1)
            |> Nx.transpose()
    
    # Extract the raw, rounded output to match the NumPy output
    neural_network(input, weights)
    |> Nx.to_flat_list
    |> List.first
    |> Float.round(2)
    |> tap(&IO.puts/1)
  end

end
