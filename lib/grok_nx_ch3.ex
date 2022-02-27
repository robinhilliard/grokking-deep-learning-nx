defmodule GrokNxCh3 do
  @moduledoc """
  Andrew W. Trask. “Grokking Deep Learning.” Manning Press
  Chapter 3 examples ported to Nx
  
  """
  
  import Nx.Defn
  import Nx, only: :sigils

  # Trivial, but idiomatic to Nx for this sort of numerically
  # optimisable function to be a defn not a def.
  defn neural_network_1(input, weights) do
    Nx.dot(input, weights)
  end
  
  def example_1() do
    # This is the first example ported, based on the first NumPy
    # example towards the end of Chapter 3. Makes use of the vector
    # and matrix sigils.
    
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
    neural_network_1(input, weights)
    |> Nx.to_flat_list
    |> List.first
    |> Float.round(2)
    |> tap(&IO.puts/1)
  end
  
  
  defn neural_network_2(input, weights) do
    input
    |> Nx.dot(weights[0])
    |> Nx.dot(weights[1])
  end
  
  
  def example_2() do
    # Stacked network with a single hidden layer.
    # I may be overdoing it trying to make this version
    # look close to the way the original NumPy version
    # assembles the weights. I didn't use the sigils
    # in this example to keep the original comment format
    # in the weight tensors. We have to make our two
    # weights matrices into tensors straight away to be
    # able to transpose them.
    
    # input -> hidden
    ih_wgt = Nx.tensor([
      [0.1, 0.2, -0.1], # hid[0]
      [-0.1, 0.1, 0.9], # hid[1]
      [0.1, 0.4, 0.1]   # hid[2]
    ]) |> Nx.transpose  # .T in NumPy example

    # hidden -> prediction
    hp_wgt = Nx.tensor([
      [0.3, 1.1, -0.3], # hurt?
      [0.1, 0.2, 0.0],  # win?
      [0.0, 1.3, 0.1]   # sad?
    ]) |> Nx.transpose  # .T in NumPy example

    # Create 0s in the target shape, then paste the two
    # weights matrices into position. They have to be
    # 3D not 2D to use put_slice so we add an extra top
    # layer wrapper dimension. Interesting to work out but
    # I sort of hope there is a better way to do this.
    weights = Nx.broadcast(0.0, {2, 3, 3})
    |> Nx.put_slice([0, 0, 0], ih_wgt |> Nx.new_axis(0))
    |> Nx.put_slice([1, 0, 0], hp_wgt |> Nx.new_axis(0))
    
    # From here we repeat code from example 1 but format
    # to show the three predictions
    
    toes_wlrec_nfans = ~M[
        8.5  9.5 9.9 9.0
        0.65 0.8 0.8 0.9
        1.2  1.3 0.5 1.0]
    
    input = toes_wlrec_nfans
        |> Nx.slice_axis(0, 1, 1)
        |> Nx.transpose()
    
    neural_network_2(input, weights)
    |> Nx.to_flat_list
    |> Enum.map(&Float.round(&1, 3))
    |> tap(&IO.inspect/1)
    
  end

end
