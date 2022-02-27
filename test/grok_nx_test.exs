defmodule GrokNxTest do
  use ExUnit.Case

  test "chapter_3_example_1" do
    assert GrokNxCh3Ex1.example() == 0.98
  end
  
  test "chapter_3_example_2" do
    pred = GrokNxCh3Ex2.example()
    assert ^pred = [0.213, 0.145, 0.506]
  end
  
end
