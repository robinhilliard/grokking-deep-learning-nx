defmodule GrokNxCh3Ex2 do
  @moduledoc """
  Andrew W. Trask. “Grokking Deep Learning.” Manning Press
  Chapter 3 example 2
  Stacked network with a single hidden layer.
  I may be overdoing it trying to make this version
  look close to the way the original NumPy version
  assembles the weights. I didn't use the sigils
  in this example to keep the original comment format
  in the weight tensors. We have to make our two
  weights matrices into tensors straight away to be
  able to transpose them.
  """
  
  
  import Nx.Defn
  import Nx, only: :sigils

  
  defn neural_network(input, weights) do
    input
    |> Nx.dot(weights[0])
    |> Nx.dot(weights[1])
  end
  
  
  def example() do
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
    
    neural_network(input, weights)
    |> Nx.to_flat_list
    |> Enum.map(&Float.round(&1, 3))
    |> tap(&IO.inspect/1)
    
  end

end
