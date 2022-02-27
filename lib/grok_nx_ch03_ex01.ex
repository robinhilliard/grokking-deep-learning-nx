defmodule GrokNxCh3Ex1 do
  @moduledoc """
  Andrew W. Trask. “Grokking Deep Learning.” Manning Press
  Chapter 3 example 1
  This is the first example ported, based on the first NumPy
  example towards the end of Chapter 3. Makes use of the vector
  and matrix sigils.
  """
  
  
  import Nx.Defn
  import Nx, only: :sigils

  
  # Trivial, but idiomatic to Nx for this sort of numerically
  # optimisable function to be a defn not a def.
  defn neural_network(input, weights) do
    Nx.dot(input, weights)
  end
  
  
  def example() do
    weights = ~V[0.1 0.2 0.0]
    
    # toes = number of pitches/bowls
    # wlrec = win/loss record
    # nfans = number of fans
    # See below for why this was combined into a single tensor.
    toes_wlrec_nfans = ~M[
        8.5  9.5 9.9 9.0
        0.65 0.8 0.8 0.9
        1.2  1.3 0.5 1.0] # "]" has to be on this line
    
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
